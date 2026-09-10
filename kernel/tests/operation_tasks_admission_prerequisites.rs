#![cfg(feature = "operation-tasks")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel::expressions::{ColumnName, Scalar};
use delta_kernel::plans::ir::nodes::{DynamicScan, FileType, Operator};
use delta_kernel::schema::{DataType, StructField, StructType, ToSchema};
use delta_kernel::tasks::{AdmittedPlan, PlanAdmissionError, Resource, TaskLimits};
use delta_kernel::FileMeta;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
    static ALLOCATED_BYTES: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn count_allocation(bytes: usize) {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
    let _ = ALLOCATED_BYTES.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n.checked_add(bytes).unwrap()));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn dynamic_scan_revalidation_does_not_allocate_column_path_vectors() {
    for depth in [0, 3, 32] {
        let mut schema = StructType::try_new([
            StructField::not_null("path", DataType::STRING),
            StructField::not_null("size", DataType::LONG),
            StructField::not_null("modified", DataType::LONG),
            StructField::nullable("dv", DeletionVectorDescriptor::to_schema()),
        ])
        .unwrap();
        for _ in 0..depth {
            schema = StructType::try_new([StructField::not_null("nested", schema)]).unwrap();
        }
        let input = Arc::new(schema);
        let column = |leaf: &str| {
            ColumnName::new(std::iter::repeat_n("nested", depth).chain(std::iter::once(leaf)))
        };
        // Construction initializes the shared DV schema outside the measured revalidation.
        let scan = DynamicScan::try_new(
            &input,
            Arc::new(StructType::try_new([]).unwrap()),
            FileType::Parquet,
            "memory:///".parse().unwrap(),
            std::iter::empty::<String>(),
            column("path"),
            column("size"),
            column("modified"),
            column("dv"),
        )
        .unwrap();
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let result = scan.validate_input(&input);
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        result.unwrap();
        assert_eq!(allocations, 0, "path depth {depth}");
        assert_plan_shape_bound(delta_kernel::plans::ir::plan::Plan {
            nodes: vec![
                delta_kernel::plans::ir::plan::PlanNode::new(
                    delta_kernel::plans::ir::nodes::Values::new(Arc::clone(&input), vec![]),
                    vec![],
                ),
                delta_kernel::plans::ir::plan::PlanNode::new(scan, vec![0]),
            ],
        });
    }
}

#[test]
fn pinned_url_clone_allocates_only_its_canonical_serialization() {
    let mut urls: Vec<String> = [
        "https://example.com/path",
        "http://127.0.0.1/path",
        "http://[::1]/path",
        "file:///tmp/part.parquet",
        "memory:///table/",
        "mailto:user@example.com",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let prefix = "memory:///";
    urls.push(format!("{prefix}{}", "x".repeat((2 << 20) - prefix.len())));
    let spare = "x".repeat(1 << 20);
    for url in urls {
        for fragment in [false, true] {
            let mut file = FileMeta {
                location: url.parse().unwrap(),
                last_modified: 0,
                size: 0,
            };
            if fragment {
                file.location.set_fragment(Some(&spare));
                file.location.set_fragment(None);
            } else {
                file.location.set_query(Some(&spare));
                file.location.set_query(None);
            }
            let length = file.location.as_str().len();
            ALLOCATIONS.with(|count| count.set(Some(0)));
            let cloned = file.location.clone();
            let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
            let cloned = String::from(cloned);
            let source = String::from(file.location);
            assert!(source.capacity() > length);
            assert_eq!(allocations, 1);
            assert_eq!(cloned.capacity(), length);
            assert_eq!(cloned, source);
        }
    }
}

#[test]
fn plan_shape_checks_topology_and_budget_before_retaining_a_plan() {
    use delta_kernel::plans::ir::nodes::{UnionAll, Values};
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::tasks::{PlanShapeError, Resource, TaskLimits};

    let schema = Arc::new(StructType::try_new([]).unwrap());
    let source = || PlanNode::new(Values::new(Arc::clone(&schema), vec![]), vec![]);
    let mut plan = Plan {
        nodes: vec![source()],
    };
    let limits = TaskLimits::qualification();
    let shape = check_plan_shape(&plan, &limits).unwrap();
    assert_eq!(shape.nodes(), 1);
    assert_eq!(shape.depth(), 1);
    let exact = limits.with_limit(Resource::PlanEncodedBytes, shape.encoded_bytes());
    assert!(check_plan_shape(&plan, &exact).is_ok());
    assert!(matches!(
        check_plan_shape(&plan, &exact.with_limit(Resource::PlanEncodedBytes, shape.encoded_bytes() - 1)),
        Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::PlanEncodedBytes
    ));
    plan.nodes.push(PlanNode::new(UnionAll, vec![0]));
    assert_eq!(check_plan_shape(&plan, &limits).unwrap().depth(), 2);
    plan.nodes[1].inputs[0] = 1;
    assert_eq!(
        check_plan_shape(&plan, &limits),
        Err(PlanShapeError::InvalidInput { node: 1 })
    );
    plan.nodes[1].inputs.clear();
    assert_eq!(
        check_plan_shape(&plan, &limits),
        Err(PlanShapeError::InvalidArity { node: 1 })
    );
    assert_eq!(
        check_plan_shape(&Plan { nodes: vec![] }, &limits),
        Err(PlanShapeError::Empty)
    );
}

fn assert_plan_shape_bound(plan: delta_kernel::plans::ir::plan::Plan) {
    use delta_kernel::plans::Operation;
    use delta_kernel::tasks::TaskLimits;

    ALLOCATIONS.with(|count| count.set(Some(0)));
    let shape = check_plan_shape(&plan, &TaskLimits::qualification());
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    let shape = shape.unwrap();
    assert_eq!(allocations, 0, "borrowed preflight does not allocate");
    let encoded = Operation::QueryPlan(plan).to_proto_bytes();
    assert!(
        shape.encoded_bytes() >= encoded.len(),
        "{shape:?}, actual {}",
        encoded.len()
    );
}

fn expression_plan(
    expr: delta_kernel::expressions::Expression,
) -> delta_kernel::plans::ir::plan::Plan {
    use delta_kernel::plans::ir::nodes::{Project, Values};
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};

    let schema = Arc::new(StructType::try_new([]).unwrap());
    Plan {
        nodes: vec![
            PlanNode::new(Values::new(Arc::clone(&schema), vec![]), vec![]),
            PlanNode::new(
                Project {
                    expr: Arc::new(expr),
                    schema,
                },
                vec![0],
            ),
        ],
    }
}

#[test]
fn plan_shape_handles_nested_scalars_schemas_and_all_operators() {
    use delta_kernel::expressions::{ArrayData, DecimalData, MapData, Scalar, StructData};
    use delta_kernel::plans::ir::nodes::{
        Agg, Aggregate, Filter, NonNullByOperands, ScanFile, ScanJson, ScanParquet, SemiJoin,
        UnionAll, Values,
    };
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::{ArrayType, DecimalType, MapType, MetadataValue};

    let mut field = StructField::nullable("text\0\n雪", DataType::STRING).with_metadata([
        ("number", MetadataValue::Number(i64::MIN)),
        ("string", MetadataValue::String("x".repeat(1 << 16))),
        ("bool", MetadataValue::Boolean(false)),
        (
            "json",
            MetadataValue::Other(
                serde_json::json!({"\u{0000}\n雪": [null, true, -1.25e100, {"x": "\"\\\t"}]}),
            ),
        ),
    ]);
    assert_unadmitted_map(&Plan {
        nodes: vec![PlanNode::new(
            Values::new(
                Arc::new(StructType::try_new([field.clone()]).unwrap()),
                vec![],
            ),
            vec![],
        )],
    });
    field.metadata.clear();
    let nested = StructType::try_new([field.clone()]).unwrap();
    let values = vec![
        Scalar::Integer(i32::MIN),
        Scalar::Long(i64::MIN),
        Scalar::Short(i16::MIN),
        Scalar::Byte(i8::MIN),
        Scalar::Float(f32::NAN),
        Scalar::Double(f64::INFINITY),
        Scalar::String("abc雪".into()),
        Scalar::Boolean(false),
        Scalar::Timestamp(i64::MIN),
        Scalar::TimestampNtz(i64::MAX),
        Scalar::Date(i32::MIN),
        Scalar::Binary(vec![255; 129]),
        Scalar::IntervalYearMonth(-1),
        Scalar::IntervalDayTime(-1),
        Scalar::Decimal(DecimalData::try_new(-123, DecimalType::try_new(38, 10).unwrap()).unwrap()),
        Scalar::Null(DataType::from(nested.clone())),
        Scalar::Null(DataType::unshredded_variant()),
        Scalar::Struct(
            StructData::try_new(vec![field.clone()], vec![Scalar::String("value".into())]).unwrap(),
        ),
        Scalar::Array(
            ArrayData::try_new(
                ArrayType::new(DataType::STRING, true),
                [Scalar::String("a".into()), Scalar::Null(DataType::STRING)],
            )
            .unwrap(),
        ),
        Scalar::Map(
            MapData::try_new(
                MapType::new(DataType::STRING, DataType::LONG, true),
                [(Scalar::String("k".into()), Scalar::Long(-1))],
            )
            .unwrap(),
        ),
    ];
    let schema = Arc::new(
        StructType::try_new(
            values
                .iter()
                .enumerate()
                .map(|(i, v)| StructField::nullable(i.to_string(), v.data_type())),
        )
        .unwrap(),
    );
    let source = PlanNode::new(
        Values::new(Arc::clone(&schema), vec![values.clone()]),
        vec![],
    );
    let file = ScanFile {
        meta: FileMeta {
            location: "memory:///table/file".parse().unwrap(),
            size: u64::MAX,
            last_modified: i64::MIN,
        },
        file_constants: values,
    };
    let mut nodes = vec![source];
    nodes.push(PlanNode::new(
        ScanJson {
            files: vec![file.clone()],
            file_constant_columns: vec!["constant".into()],
            schema: Arc::clone(&schema),
        },
        vec![],
    ));
    nodes.push(PlanNode::new(
        ScanParquet {
            files: vec![file],
            file_constant_columns: vec!["constant".into()],
            schema: Arc::clone(&schema),
        },
        vec![],
    ));
    nodes.push(PlanNode::new(
        Filter {
            predicate: Arc::new(delta_kernel::expressions::Predicate::TRUE),
        },
        vec![0],
    ));
    nodes.push(PlanNode::new(UnionAll, vec![1, 2, 3]));
    let c = || ColumnName::new(["0"]);
    let operands = || NonNullByOperands {
        value: c(),
        null_sentinel: c(),
        key: c(),
    };
    nodes.push(PlanNode::new(
        Aggregate {
            group_by: vec![c()],
            aggs: vec![
                Agg::Min(c()),
                Agg::Max(c()),
                Agg::Sum(c()),
                Agg::Count(c()),
                Agg::CountStar,
                Agg::MinNonNullBy(operands()),
                Agg::MaxNonNullBy(operands()),
            ],
            schema,
        },
        vec![4],
    ));
    nodes.push(PlanNode::new(
        SemiJoin {
            inverted: true,
            probe_keys: vec![c()],
            build_keys: vec![c()],
        },
        vec![4, 5],
    ));
    assert_plan_shape_bound(Plan { nodes });
}

fn expression_wrappers(
    expr: delta_kernel::expressions::Expression,
) -> Vec<delta_kernel::expressions::Expression> {
    use delta_kernel::expressions::*;
    vec![
        expr.clone(),
        Expression::Struct(vec![Arc::new(expr.clone())], None),
        Expression::Struct(vec![], Some(Arc::new(expr.clone()))),
        Expression::Unary(UnaryExpression {
            op: UnaryExpressionOp::ToJson,
            expr: Box::new(expr.clone()),
        }),
        Expression::Binary(BinaryExpression {
            op: BinaryExpressionOp::Plus,
            left: Box::new(expr.clone()),
            right: Box::new(lit(1)),
        }),
        Expression::Variadic(VariadicExpression {
            op: VariadicExpressionOp::Coalesce,
            exprs: vec![expr.clone()],
        }),
        Expression::ParseJson(ParseJsonExpression {
            json_expr: Box::new(expr.clone()),
            output_schema: Arc::new(StructType::try_new([]).unwrap()),
        }),
        Expression::MapToStruct(MapToStructExpression {
            map_expr: Box::new(expr.clone()),
        }),
        Expression::Cast(CastExpression {
            expr: Box::new(expr.clone()),
            target: DataType::from(delta_kernel::schema::ArrayType::new(DataType::STRING, true)),
        }),
        Expression::StructPatch(ExpressionStructPatch {
            input_path: Some(ColumnName::new(["nested"])),
            field_patches: [(
                "field".into(),
                ExpressionFieldPatch {
                    keep_input: true,
                    optional: true,
                    insertions: vec![Arc::new(expr.clone())],
                },
            )]
            .into(),
            ..Default::default()
        }),
        Expression::StructPatch(ExpressionStructPatch {
            prepended_fields: vec![Arc::new(expr.clone())],
            ..Default::default()
        }),
        Expression::StructPatch(ExpressionStructPatch {
            appended_fields: vec![Arc::new(expr.clone())],
            ..Default::default()
        }),
        Expression::Predicate(Box::new(Predicate::BooleanExpression(expr.clone()))),
        Expression::Predicate(Box::new(Predicate::Not(Box::new(
            Predicate::BooleanExpression(expr.clone()),
        )))),
        Expression::Predicate(Box::new(Predicate::Unary(UnaryPredicate {
            op: UnaryPredicateOp::IsNull,
            expr: Box::new(expr.clone()),
        }))),
        Expression::Predicate(Box::new(Predicate::Binary(BinaryPredicate {
            op: BinaryPredicateOp::Equal,
            left: Box::new(lit(1)),
            right: Box::new(expr.clone()),
        }))),
        Expression::Predicate(Box::new(Predicate::Junction(JunctionPredicate {
            op: JunctionPredicateOp::And,
            preds: vec![Predicate::BooleanExpression(expr)],
        }))),
    ]
}

#[test]
fn plan_shape_walks_all_expression_positions_and_rejects_nested_unknowns() {
    use delta_kernel::expressions::{Expression, Predicate};
    use delta_kernel::tasks::TaskLimits;

    for expr in expression_wrappers(Expression::Column(ColumnName::new(["field", "nested"]))) {
        let plan = expression_plan(expr.clone());
        if expression_has_unadmitted_patch(&expr) {
            assert_unadmitted_map(&plan);
        } else {
            assert_plan_shape_bound(plan);
        }
    }
    for unknown in [
        Expression::Unknown("hidden".into()),
        Expression::Predicate(Box::new(Predicate::Unknown("hidden".into()))),
    ] {
        for expr in expression_wrappers(unknown) {
            let expected = expression_rejection(&expr);
            let plan = expression_plan(expr);
            ALLOCATIONS.with(|count| count.set(Some(0)));
            let result = check_plan_shape(&plan, &TaskLimits::qualification());
            let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
            assert_eq!(result, Err(expected));
            assert_eq!(
                allocations, 0,
                "failed borrowed preflight does not allocate"
            );
        }
    }
}

#[test]
fn plan_shape_rejects_each_resource_boundary_without_unbounded_scratch() {
    use delta_kernel::expressions::{lit, Expression, ExpressionFieldPatch, ExpressionStructPatch};
    use delta_kernel::tasks::{PlanShapeError, Resource, TaskLimits};

    let plan = expression_plan(lit(1));
    let limits = TaskLimits::qualification();
    let shape = check_plan_shape(&plan, &limits).unwrap();
    for (resource, limit) in [
        (Resource::PlanNodes, 1),
        (Resource::PlanDepth, 1),
        (Resource::PlanEncodedBytes, shape.encoded_bytes() - 1),
        (Resource::WorkUnits, shape.work_units() - 1),
        (Resource::SchemaNodes, 0),
        (Resource::SchemaDepth, 0),
    ] {
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let result = check_plan_shape(&plan, &limits.with_limit(resource, limit));
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert!(
            matches!(result, Err(PlanShapeError::ResourceExhausted(e)) if e.resource == resource),
            "{resource:?}: {result:?}"
        );
        assert_eq!(allocations, 0);
    }
    let mut expr = lit(1);
    for _ in 0..65 {
        expr = Expression::Struct(vec![Arc::new(expr)], None);
    }
    let deep = expression_plan(expr);
    assert!(
        matches!(check_plan_shape(&deep, &limits.with_limit(Resource::SchemaDepth, usize::MAX)), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::SchemaDepth && e.limit == 64)
    );
    let mut patch = ExpressionStructPatch::default();
    patch.field_patches.reserve(4096);
    patch
        .field_patches
        .insert("x".into(), ExpressionFieldPatch::default());
    let sparse = expression_plan(Expression::StructPatch(patch));
    assert!(matches!(
        check_plan_shape(&sparse, &limits.with_limit(Resource::WorkUnits, 100)),
        Err(PlanShapeError::UnadmittedMap)
    ));
}

#[derive(Debug, PartialEq)]
struct NeverInvoke;

impl delta_kernel::expressions::OpaqueExpressionOp for NeverInvoke {
    fn name(&self) -> &str {
        panic!("preflight must not invoke an opaque callback")
    }
    fn eval_expr_scalar(
        &self,
        _: &delta_kernel::expressions::ScalarExpressionEvaluator<'_>,
        _: &[delta_kernel::expressions::Expression],
    ) -> delta_kernel::DeltaResult<delta_kernel::expressions::Scalar> {
        panic!("preflight must not invoke an opaque callback")
    }
}

#[test]
fn plan_shape_rejects_opaque_callbacks_without_invoking_or_retaining_them() {
    use delta_kernel::expressions::{Expression, OpaqueExpression};
    use delta_kernel::tasks::TaskLimits;

    let op = Arc::new(NeverInvoke);
    let expr = Expression::Opaque(OpaqueExpression {
        op: op.clone(),
        exprs: vec![],
    });
    for expr in expression_wrappers(expr) {
        let expected = expression_rejection(&expr);
        let plan = expression_plan(expr);
        let before = Arc::strong_count(&op);
        assert_eq!(
            check_plan_shape(&plan, &TaskLimits::qualification()),
            Err(expected)
        );
        assert_eq!(Arc::strong_count(&op), before);
    }
    assert_eq!(Arc::strong_count(&op), 1);
}

#[test]
fn plan_shape_topology_and_schema_depth_limits_are_independent() {
    use delta_kernel::expressions::Predicate;
    use delta_kernel::plans::ir::nodes::{Filter, Values};
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::tasks::{PlanShapeError, Resource, TaskLimits};

    let mut schema =
        StructType::try_new([StructField::nullable("leaf", DataType::STRING)]).unwrap();
    for _ in 1..64 {
        schema = StructType::try_new([StructField::nullable("nested", schema)]).unwrap();
    }
    let mut plan = Plan {
        nodes: vec![PlanNode::new(Values::new(Arc::new(schema), vec![]), vec![])],
    };
    let limits = TaskLimits::qualification();
    assert_plan_shape_bound(plan.clone());
    assert!(
        matches!(check_plan_shape(&plan, &limits.with_limit(Resource::SchemaDepth, 63)), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::SchemaDepth)
    );
    for input in 0..63 {
        plan.nodes.push(PlanNode::new(
            Filter {
                predicate: Arc::new(Predicate::TRUE),
            },
            vec![input],
        ));
    }
    assert_eq!(check_plan_shape(&plan, &limits).unwrap().depth(), 64);
    plan.nodes.push(PlanNode::new(
        Filter {
            predicate: Arc::new(Predicate::TRUE),
        },
        vec![63],
    ));
    assert!(
        matches!(check_plan_shape(&plan, &limits), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::PlanDepth)
    );
}

fn check_plan_shape(
    plan: &delta_kernel::plans::ir::plan::Plan,
    limits: &delta_kernel::tasks::TaskLimits,
) -> Result<delta_kernel::tasks::PlanShape, delta_kernel::tasks::PlanShapeError> {
    let mut scratch = [0; 4096];
    delta_kernel::tasks::PlanShape::check(plan, limits, &mut scratch)
}

#[test]
fn plan_shape_borrows_only_the_required_scratch_prefix() {
    use delta_kernel::expressions::lit;
    use delta_kernel::tasks::{PlanShape, PlanShapeError, TaskLimits};
    let plan = expression_plan(lit(1));
    let limits = TaskLimits::qualification();
    let mut too_short = [42];
    assert_eq!(
        PlanShape::check(&plan, &limits, &mut too_short),
        Err(PlanShapeError::ScratchTooSmall { required: 2 })
    );
    assert_eq!(too_short, [42]);
    let mut scratch = [42; 4];
    let shape = PlanShape::check(&plan, &limits, &mut scratch).unwrap();
    assert_eq!(shape.depth(), 2);
    assert_eq!(scratch, [1, 2, 42, 42]);
}

#[test]
fn plan_shape_bounds_repeated_decimal_and_escaping_heavy_metadata() {
    use delta_kernel::expressions::{DecimalData, Scalar};
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::{DecimalType, MetadataValue};
    let decimal =
        Scalar::Decimal(DecimalData::try_new(-123, DecimalType::try_new(38, 10).unwrap()).unwrap());
    let schema =
        Arc::new(StructType::try_new([StructField::nullable("d", decimal.data_type())]).unwrap());
    assert_plan_shape_bound(Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![decimal]; 4096]),
            vec![],
        )],
    });
    let metadata = serde_json::Value::String("\u{0000}".repeat(16384));
    let schema = Arc::new(
        StructType::try_new([StructField::nullable("m", DataType::STRING)
            .with_metadata([("control", MetadataValue::Other(metadata))])])
        .unwrap(),
    );
    assert_unadmitted_map(&Plan {
        nodes: vec![PlanNode::new(Values::new(schema, vec![]), vec![])],
    });
}

#[test]
fn plan_shape_charges_cast_display_work_as_well_as_its_retained_target() {
    use delta_kernel::expressions::{lit, CastExpression, Expression, Scalar};
    use delta_kernel::tasks::{PlanShapeError, Resource, TaskLimits};
    let target = DataType::from(
        StructType::try_new(
            (0..128).map(|i| StructField::nullable(format!("field_{i}"), DataType::STRING)),
        )
        .unwrap(),
    );
    let limits = TaskLimits::qualification();
    let null = expression_plan(Expression::Literal(Scalar::Null(target.clone())));
    let baseline = check_plan_shape(&null, &limits).unwrap();
    let cast = expression_plan(Expression::Cast(CastExpression {
        expr: Box::new(lit(1)),
        target,
    }));
    let shape = check_plan_shape(&cast, &limits).unwrap();
    // The cast retains one more IR node than the typed NULL; formatting visits add more work.
    assert!(shape.work_units() > baseline.work_units() + 1);
    assert!(check_plan_shape(
        &cast,
        &limits.with_limit(Resource::WorkUnits, shape.work_units())
    )
    .is_ok());
    assert!(
        matches!(check_plan_shape(&cast, &limits.with_limit(Resource::WorkUnits, shape.work_units() - 1)), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::WorkUnits)
    );
}

impl delta_kernel::expressions::OpaquePredicateOp for NeverInvoke {
    fn name(&self) -> &str {
        panic!("preflight must not invoke an opaque callback")
    }
    fn eval_pred_scalar(
        &self,
        _: &delta_kernel::expressions::ScalarExpressionEvaluator<'_>,
        _: &delta_kernel::kernel_predicates::DirectPredicateEvaluator<'_>,
        _: &[delta_kernel::expressions::Expression],
        _: bool,
    ) -> delta_kernel::DeltaResult<Option<bool>> {
        panic!("preflight must not invoke an opaque callback")
    }
    fn eval_as_data_skipping_predicate(
        &self,
        _: &delta_kernel::kernel_predicates::DirectDataSkippingPredicateEvaluator<'_>,
        _: &[delta_kernel::expressions::Expression],
        _: bool,
    ) -> Option<bool> {
        panic!("preflight must not invoke an opaque callback")
    }
    fn as_data_skipping_predicate(
        &self,
        _: &delta_kernel::kernel_predicates::IndirectDataSkippingPredicateEvaluator<'_>,
        _: &[delta_kernel::expressions::Expression],
        _: bool,
    ) -> Option<delta_kernel::expressions::Predicate> {
        panic!("preflight must not invoke an opaque callback")
    }
}

#[test]
fn plan_shape_rejects_nested_opaque_predicates_without_calling_them() {
    use delta_kernel::expressions::{Expression, OpaquePredicate, Predicate};
    use delta_kernel::tasks::TaskLimits;
    let op = Arc::new(NeverInvoke);
    let expr = Expression::Predicate(Box::new(Predicate::Opaque(OpaquePredicate {
        op: op.clone(),
        exprs: vec![],
    })));
    for expr in expression_wrappers(expr) {
        let expected = expression_rejection(&expr);
        assert_eq!(
            check_plan_shape(&expression_plan(expr), &TaskLimits::qualification()),
            Err(expected)
        );
    }
    assert_eq!(Arc::strong_count(&op), 1);
}

fn check_literal_plan(
    plan: &delta_kernel::plans::ir::plan::Plan,
    limits: &delta_kernel::tasks::TaskLimits,
) -> Result<delta_kernel::tasks::PlanShape, delta_kernel::tasks::PlanShapeError> {
    let mut scratch = [0; 4096];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = delta_kernel::tasks::PlanShape::check_literals(plan, limits, &mut scratch);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(allocations, 0, "literal preflight must not allocate");
    result
}

#[test]
fn literal_preflight_rejects_values_width_type_and_nullability_mismatches() {
    use delta_kernel::expressions::Scalar;
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let schema =
        Arc::new(StructType::try_new([StructField::not_null("x", DataType::LONG)]).unwrap());
    for row in [
        vec![],
        vec![Scalar::Long(1), Scalar::Long(2)],
        vec![Scalar::Integer(1)],
        vec![Scalar::Null(DataType::LONG)],
    ] {
        let plan = Plan {
            nodes: vec![PlanNode::new(
                Values::new(Arc::clone(&schema), vec![row]),
                vec![],
            )],
        };
        // Structural measurement remains independent of literal compatibility.
        check_plan_shape(&plan, &TaskLimits::qualification()).unwrap();
        assert_eq!(
            check_literal_plan(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::InvalidLiteral)
        );
    }
    let plan = Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![Scalar::Long(1)]]),
            vec![],
        )],
    };
    check_literal_plan(&plan, &TaskLimits::qualification()).unwrap();
}

#[test]
fn literal_preflight_rejects_deserialized_container_invariant_violations() {
    use delta_kernel::expressions::{ArrayData, Expression, MapData, Scalar, StructData};
    use delta_kernel::schema::{ArrayType, MapType};
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let array = Scalar::Array(
        ArrayData::try_new(ArrayType::new(DataType::LONG, false), [Scalar::Long(1)]).unwrap(),
    );
    let map = Scalar::Map(
        MapData::try_new(
            MapType::new(DataType::STRING, DataType::LONG, false),
            [(Scalar::String("k".into()), Scalar::Long(1))],
        )
        .unwrap(),
    );
    let structure = Scalar::Struct(
        StructData::try_new(
            vec![StructField::not_null("x", DataType::LONG)],
            vec![Scalar::Long(1)],
        )
        .unwrap(),
    );
    let mut malformed = Vec::new();
    for replacement in [Scalar::Integer(1), Scalar::Null(DataType::LONG)] {
        let mut json = serde_json::to_value(&array).unwrap();
        json["Array"]["elements"][0] = serde_json::to_value(&replacement).unwrap();
        malformed.push(json);
        let mut json = serde_json::to_value(&structure).unwrap();
        json["Struct"]["values"][0] = serde_json::to_value(&replacement).unwrap();
        malformed.push(json);
        let mut json = serde_json::to_value(&map).unwrap();
        json["Map"]["pairs"][0][1] = serde_json::to_value(&replacement).unwrap();
        malformed.push(json);
    }
    let mut json = serde_json::to_value(&map).unwrap();
    json["Map"]["pairs"][0][0] = serde_json::to_value(Scalar::Null(DataType::STRING)).unwrap();
    malformed.push(json);
    let mut json = serde_json::to_value(&structure).unwrap();
    json["Struct"]["values"] = serde_json::json!([]);
    malformed.push(json);
    for json in malformed {
        let value: Scalar = serde_json::from_value(json).unwrap();
        let plan = expression_plan(Expression::Literal(value));
        assert_eq!(
            check_literal_plan(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::InvalidLiteral)
        );
    }
}

#[test]
fn literal_preflight_rejects_deserialized_decimal_precision_and_scale() {
    use delta_kernel::expressions::{Expression, Scalar};
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let valid = Scalar::decimal(999, 3, 0).unwrap();
    for (precision, scale, bits) in [
        (0, 0, 0),
        (39, 0, 0),
        (3, 4, 1),
        (3, 0, 1000),
        (38, 0, i128::MIN),
    ] {
        let mut json = serde_json::to_value(&valid).unwrap();
        json["Decimal"]["ty"]["precision"] = precision.into();
        json["Decimal"]["ty"]["scale"] = scale.into();
        // serde_json cannot represent all i128 values through Value; mutate serialized text.
        let text = serde_json::to_string(&json)
            .unwrap()
            .replace("999", &bits.to_string());
        let value: Scalar = serde_json::from_str(&text).unwrap();
        let plan = expression_plan(Expression::Literal(value));
        assert_eq!(
            check_literal_plan(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::InvalidLiteral)
        );
    }
    check_literal_plan(
        &expression_plan(Expression::Literal(valid)),
        &TaskLimits::qualification(),
    )
    .unwrap();
}

#[test]
fn literal_preflight_checks_scan_constant_names_width_types_and_metadata() {
    use delta_kernel::expressions::Scalar;
    use delta_kernel::plans::ir::nodes::{ScanFile, ScanJson, ScanParquet};
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::MetadataColumnSpec;
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let schema = Arc::new(
        StructType::try_new([
            StructField::not_null("x", DataType::LONG),
            StructField::not_null("row_index", DataType::LONG),
        ])
        .unwrap(),
    );
    for parquet in [false, true] {
        for (names, values, valid) in [
            (vec!["x"], vec![Scalar::Long(1)], true),
            (vec!["x"], vec![], false),
            (vec!["missing"], vec![Scalar::Long(1)], false),
            (
                vec!["x", "x"],
                vec![Scalar::Long(1), Scalar::Long(2)],
                false,
            ),
            (vec!["row_index"], vec![Scalar::Long(1)], false),
            (vec!["x"], vec![Scalar::Integer(1)], false),
            (vec!["x"], vec![Scalar::Null(DataType::LONG)], false),
        ] {
            let has_metadata = names.as_slice() == ["row_index"];
            let schema = if has_metadata {
                Arc::new(
                    StructType::try_new([
                        StructField::not_null("x", DataType::LONG),
                        StructField::create_metadata_column(
                            "row_index",
                            MetadataColumnSpec::RowIndex,
                        ),
                    ])
                    .unwrap(),
                )
            } else {
                Arc::clone(&schema)
            };
            let files = vec![ScanFile {
                meta: FileMeta {
                    location: "memory:///file".parse().unwrap(),
                    size: 1,
                    last_modified: 0,
                },
                file_constants: values,
            }];
            let file_constant_columns = names.into_iter().map(str::to_string).collect();
            let node = if parquet {
                PlanNode::new(
                    ScanParquet {
                        files,
                        file_constant_columns,
                        schema: Arc::clone(&schema),
                    },
                    vec![],
                )
            } else {
                PlanNode::new(
                    ScanJson {
                        files,
                        file_constant_columns,
                        schema: Arc::clone(&schema),
                    },
                    vec![],
                )
            };
            let result =
                check_literal_plan(&Plan { nodes: vec![node] }, &TaskLimits::qualification());
            if valid {
                result.unwrap();
            } else {
                assert_eq!(
                    result,
                    Err(if has_metadata {
                        PlanShapeError::UnadmittedMap
                    } else {
                        PlanShapeError::InvalidLiteral
                    })
                );
            }
        }
    }
}

#[test]
fn literal_preflight_preserves_nested_field_order() {
    use delta_kernel::expressions::{Scalar, StructData};
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::{ArrayType, MetadataValue};
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let a = StructField::nullable("a", DataType::LONG);
    let b = StructField::nullable("b", DataType::LONG);
    let expected = StructType::try_new([a.clone(), b.clone()]).unwrap();
    let outer =
        Arc::new(StructType::try_new([StructField::nullable("s", expected.clone())]).unwrap());
    let ordered = Scalar::Struct(
        StructData::try_new(
            vec![a.clone(), b.clone()],
            vec![Scalar::Long(1), Scalar::Long(2)],
        )
        .unwrap(),
    );
    let reversed = Scalar::Struct(
        StructData::try_new(
            vec![b.clone(), a.clone()],
            vec![Scalar::Long(2), Scalar::Long(1)],
        )
        .unwrap(),
    );
    let mut changed = a.clone();
    changed
        .metadata
        .insert("tag".into(), MetadataValue::String("changed".into()));
    let wrong_metadata = Scalar::Struct(
        StructData::try_new(
            vec![changed, b.clone()],
            vec![Scalar::Long(1), Scalar::Long(2)],
        )
        .unwrap(),
    );
    for (value, valid) in [(ordered, true), (reversed, false)] {
        let plan = Plan {
            nodes: vec![PlanNode::new(
                Values::new(Arc::clone(&outer), vec![vec![value]]),
                vec![],
            )],
        };
        let result = check_literal_plan(&plan, &TaskLimits::qualification());
        if valid {
            result.unwrap();
        } else {
            assert_eq!(result, Err(PlanShapeError::InvalidLiteral));
        }
    }
    assert_unadmitted_map(&Plan {
        nodes: vec![PlanNode::new(
            Values::new(Arc::clone(&outer), vec![vec![wrong_metadata]]),
            vec![],
        )],
    });
    // A typed null must obey nested ordering even inside an array descriptor.
    let expected = DataType::from(ArrayType::new(expected, true));
    let reversed = DataType::from(ArrayType::new(StructType::try_new([b, a]).unwrap(), true));
    let schema = Arc::new(StructType::try_new([StructField::nullable("a", expected)]).unwrap());
    let plan = Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![Scalar::Null(reversed)]]),
            vec![],
        )],
    };
    assert_eq!(
        check_literal_plan(&plan, &TaskLimits::qualification()),
        Err(PlanShapeError::InvalidLiteral)
    );
}

#[test]
fn literal_preflight_checks_tags_duplicates_and_comparison_work_boundaries() {
    use delta_kernel::expressions::{Expression, Scalar, StructData};
    use delta_kernel::schema::{ArrayType, MapType};
    use delta_kernel::tasks::{PlanShapeError, Resource, TaskLimits};

    let mut array = ArrayType::new(DataType::LONG, true);
    array.type_name = "not-array".into();
    let mut map = MapType::new(DataType::STRING, DataType::LONG, true);
    map.type_name = "not-map".into();
    for ty in [DataType::from(array), DataType::from(map)] {
        let plan = expression_plan(Expression::Literal(Scalar::Null(ty)));
        assert_eq!(
            check_literal_plan(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::InvalidLiteral)
        );
    }
    let field = StructField::nullable("same", DataType::LONG);
    let duplicate = Scalar::Struct(
        StructData::try_new(
            vec![field.clone(), field],
            vec![Scalar::Long(1), Scalar::Long(2)],
        )
        .unwrap(),
    );
    assert_eq!(
        check_literal_plan(
            &expression_plan(Expression::Literal(duplicate)),
            &TaskLimits::qualification()
        ),
        Err(PlanShapeError::InvalidLiteral)
    );

    let literal = Scalar::Struct(
        StructData::try_new(
            vec![StructField::nullable("x", DataType::STRING)],
            vec![Scalar::String("value".into())],
        )
        .unwrap(),
    );
    let plan = expression_plan(Expression::Literal(literal));
    let limits = TaskLimits::qualification();
    let shape = check_plan_shape(&plan, &limits).unwrap();
    let literals = check_literal_plan(&plan, &limits).unwrap();
    assert_eq!(
        (shape.nodes(), shape.depth(), shape.encoded_bytes()),
        (literals.nodes(), literals.depth(), literals.encoded_bytes())
    );
    assert!(literals.work_units() > shape.work_units());
    check_literal_plan(
        &plan,
        &limits.with_limit(Resource::WorkUnits, literals.work_units()),
    )
    .unwrap();
    assert!(
        matches!(check_literal_plan(&plan, &limits.with_limit(Resource::WorkUnits, literals.work_units() - 1)), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::WorkUnits)
    );
}

#[test]
fn literal_preflight_does_not_scan_empty_reserved_metadata() {
    use delta_kernel::expressions::Scalar;
    use delta_kernel::plans::ir::nodes::{ScanFile, ScanJson};
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::tasks::TaskLimits;

    let mut extra = Vec::new();
    for reserve in [0, 8192] {
        let mut field = StructField::nullable("x", DataType::LONG);
        field.metadata.reserve(reserve);
        let capacity = field.metadata.capacity();
        let schema = Arc::new(StructType::try_new([field]).unwrap());
        let plan = Plan {
            nodes: vec![PlanNode::new(
                ScanJson {
                    schema,
                    file_constant_columns: vec!["x".into()],
                    files: vec![ScanFile {
                        meta: FileMeta {
                            location: "memory:///file".parse().unwrap(),
                            size: 1,
                            last_modified: 0,
                        },
                        file_constants: vec![Scalar::Long(1)],
                    }],
                },
                vec![],
            )],
        };
        let limits = TaskLimits::qualification();
        let shape = check_plan_shape(&plan, &limits).unwrap();
        let literals = check_literal_plan(&plan, &limits).unwrap();
        check_literal_plan(
            &plan,
            &limits.with_limit(
                delta_kernel::tasks::Resource::WorkUnits,
                literals.work_units(),
            ),
        )
        .unwrap();
        assert!(
            matches!(check_literal_plan(&plan, &limits.with_limit(delta_kernel::tasks::Resource::WorkUnits, literals.work_units() - 1)), Err(delta_kernel::tasks::PlanShapeError::ResourceExhausted(e)) if e.resource == delta_kernel::tasks::Resource::WorkUnits)
        );
        extra.push((literals.work_units() - shape.work_units(), capacity));
    }
    assert_eq!(extra[1].0, extra[0].0);
    assert!(extra[1].1 > extra[0].1);
}

#[test]
fn literal_preflight_requires_producer_admission_for_nested_metadata() {
    use delta_kernel::expressions::{Scalar, StructData};
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::MetadataValue;

    let fields: Vec<_> =
        (0..32)
            .map(|i| {
                let mut field = StructField::nullable(format!("f{i}"), DataType::LONG);
                field.metadata.insert("json".into(), MetadataValue::Other(serde_json::json!({
            "first": [true, null, {"nested": "value", "number": 1.25}], "second": [1, 2, 3]
        })));
                field
                    .metadata
                    .insert("tag".into(), MetadataValue::String("kept".into()));
                if i == 0 {
                    for key in 0..64 {
                        field
                            .metadata
                            .insert(format!("key{key}"), MetadataValue::Number(key));
                    }
                }
                field
            })
            .collect();
    let expected = StructType::try_new(fields.clone()).unwrap();
    let actual: Vec<_> = fields
        .iter()
        .cloned()
        .map(|mut field| {
            let mut metadata = std::collections::HashMap::new();
            for (key, value) in &field.metadata {
                metadata.insert(key.clone(), value.clone());
            }
            field.metadata = metadata;
            field
        })
        .collect();
    // Verify different bucket order rather than assuming insertion order changes iteration.
    assert!(fields[0].metadata.keys().ne(actual[0].metadata.keys()));
    let scalar =
        Scalar::Struct(StructData::try_new(actual, (0..32).map(Scalar::Long).collect()).unwrap());
    let schema = Arc::new(StructType::try_new([StructField::nullable("s", expected)]).unwrap());
    let plan = Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![scalar]]),
            vec![],
        )],
    };
    assert_unadmitted_map(&plan);
}

#[test]
fn hash_map_capacity_after_deletion_is_not_a_backing_allocation_bound() {
    let map = tombstone_map(delta_kernel::schema::MetadataValue::Number(1));
    assert_eq!(map.len(), 1);
}

fn tombstone_map<V: Clone>(value: V) -> std::collections::HashMap<String, V> {
    use std::collections::HashMap;
    use std::hash::BuildHasher;
    let mut map = HashMap::with_capacity(1024);
    let original = map.capacity();
    let mask = original.next_power_of_two() - 1;
    let cluster = map
        .hasher()
        .hash_one(delta_kernel::schema::ColumnMetadataKey::MetadataSpec.as_ref())
        as usize
        & mask;
    let mut keys = Vec::with_capacity(original);
    for index in 0..20_000_000 {
        let key = format!("collision-{index}");
        if map.hasher().hash_one(&key) as usize & mask == cluster {
            keys.push(key.clone());
            map.insert(key, value.clone());
            if keys.len() == original {
                break;
            }
        }
    }
    assert_eq!(map.len(), original);
    assert_eq!(map.capacity(), original);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for key in &keys[..keys.len() - 1] {
        map.remove(key);
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(allocations, 0);
    assert_eq!(map.len(), 1);
    assert!(
        map.capacity() < original / 4,
        "before={original}, after={}",
        map.capacity()
    );
    println!(
        "hash-map owner probe: original_capacity={original}, retained_capacity={}, live_entries={}",
        map.capacity(),
        map.len()
    );
    map
}

#[test]
fn borrowed_plan_rejects_unadmitted_hash_maps_before_scanning() {
    use delta_kernel::expressions::{
        Expression, ExpressionFieldPatch, ExpressionStructPatch, Scalar,
    };
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::MetadataValue;
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let mut field = StructField::nullable("x", DataType::LONG);
    field
        .metadata
        .insert("key".into(), MetadataValue::Number(1));
    let schema = Arc::new(StructType::try_new([field]).unwrap());
    let metadata = Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![Scalar::Long(1)]]),
            vec![],
        )],
    };
    let patch = expression_plan(Expression::StructPatch(ExpressionStructPatch {
        field_patches: [(
            "x".into(),
            ExpressionFieldPatch {
                keep_input: true,
                optional: false,
                insertions: vec![],
            },
        )]
        .into(),
        ..Default::default()
    }));
    for plan in [metadata, patch] {
        assert_eq!(
            check_plan_shape(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::UnadmittedMap)
        );
        assert_eq!(
            check_literal_plan(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::UnadmittedMap)
        );
    }
}

#[test]
fn borrowed_plan_rejects_nested_unadmitted_metadata() {
    use delta_kernel::expressions::{CastExpression, Expression, ParseJsonExpression, Scalar};
    use delta_kernel::schema::{ArrayType, MapType, MetadataValue};

    let nested = || {
        StructType::try_new([StructField::nullable("x", DataType::LONG)
            .with_metadata([("key", MetadataValue::Number(1))])])
        .unwrap()
    };
    let expressions = [
        Expression::Literal(Scalar::Null(DataType::from(nested()))),
        Expression::Literal(Scalar::Null(DataType::from(ArrayType::new(nested(), true)))),
        Expression::Literal(Scalar::Null(DataType::from(MapType::new(
            DataType::STRING,
            nested(),
            true,
        )))),
        Expression::ParseJson(ParseJsonExpression {
            json_expr: Box::new(delta_kernel::expressions::lit("{}")),
            output_schema: Arc::new(nested()),
        }),
        Expression::Cast(CastExpression {
            expr: Box::new(delta_kernel::expressions::lit(1)),
            target: DataType::from(nested()),
        }),
    ];
    for expression in expressions {
        assert_unadmitted_map(&expression_plan(expression));
    }
}

#[test]
fn borrowed_plan_accepts_empty_maps_with_retained_backing_without_charging_it() {
    use delta_kernel::expressions::{Expression, ExpressionStructPatch};

    let mut patch = ExpressionStructPatch::default();
    patch.field_patches.reserve(8192);
    let plan = expression_plan(Expression::StructPatch(patch));
    let limits = delta_kernel::tasks::TaskLimits::qualification();
    let shape = check_plan_shape(&plan, &limits).unwrap();
    let literal_shape = check_literal_plan(&plan, &limits).unwrap();
    assert_eq!(shape, literal_shape);
}

fn expression_has_unadmitted_patch(expr: &delta_kernel::expressions::Expression) -> bool {
    matches!(expr, delta_kernel::expressions::Expression::StructPatch(p) if !p.field_patches.is_empty())
}

fn expression_rejection(
    expr: &delta_kernel::expressions::Expression,
) -> delta_kernel::tasks::PlanShapeError {
    if expression_has_unadmitted_patch(expr) {
        delta_kernel::tasks::PlanShapeError::UnadmittedMap
    } else {
        delta_kernel::tasks::PlanShapeError::UnsupportedExpression
    }
}

fn assert_unadmitted_map(plan: &delta_kernel::plans::ir::plan::Plan) {
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = check_plan_shape(plan, &TaskLimits::qualification());
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(result, Err(PlanShapeError::UnadmittedMap));
    assert_eq!(allocations, 0);
    assert_eq!(
        check_literal_plan(plan, &TaskLimits::qualification()),
        Err(PlanShapeError::UnadmittedMap)
    );
}

#[test]
fn borrowed_plan_rejects_tombstone_metadata_without_scanning_or_lookup() {
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    let mut field = StructField::nullable("x", DataType::LONG);
    field.metadata = tombstone_map(delta_kernel::schema::MetadataValue::Number(1));
    let schema = Arc::new(StructType::try_new([field]).unwrap());
    let plan = Plan {
        nodes: vec![PlanNode::new(Values::new(schema, vec![]), vec![])],
    };
    assert_unadmitted_map(&plan);
}

#[test]
fn borrowed_plan_rejects_tombstone_field_patches_without_scanning() {
    use delta_kernel::expressions::{Expression, ExpressionFieldPatch, ExpressionStructPatch};
    let plan = expression_plan(Expression::StructPatch(ExpressionStructPatch {
        field_patches: tombstone_map(ExpressionFieldPatch::default()),
        ..Default::default()
    }));
    assert_unadmitted_map(&plan);
}

#[test]
fn admitted_fixed_width_producer_preflights_metadata_and_allocations() {
    use delta_kernel::schema::MetadataValue;
    use delta_kernel::tasks::{
        AdmittedPlan, PlanAdmissionError, PlanMetadataEntry, Resource, TaskLimits,
    };

    let text = MetadataValue::String("physical_value".into());
    let number = MetadataValue::Number(i64::MIN);
    let boolean = MetadataValue::Boolean(true);
    let metadata = [
        PlanMetadataEntry::new("physical", &text),
        PlanMetadataEntry::new("id", &number),
        PlanMetadataEntry::new("enabled", &boolean),
    ];
    let values = [i64::MIN, -1, i64::MAX];
    let limits = TaskLimits::qualification();
    ALLOCATED_BYTES.with(|count| count.set(Some(0)));
    let admitted = AdmittedPlan::try_i64_values("value", &metadata, &values, &limits).unwrap();
    let allocated_bytes = ALLOCATED_BYTES.with(|count| count.replace(None).unwrap());

    assert_eq!(admitted.shape().nodes(), 1);
    assert_eq!(admitted.shape().depth(), 1);
    assert_eq!(admitted.metadata_entries(), metadata.len());
    assert!(admitted.retained_bytes() > 0);
    assert!(admitted.metadata_bytes() > 0);
    assert!(allocated_bytes <= admitted.retained_bytes());
    assert_eq!(admitted.field_metadata("physical"), Some(&text));
    assert_eq!(admitted.field_metadata("id"), Some(&number));
    assert_eq!(admitted.field_metadata("enabled"), Some(&boolean));

    let too_small = limits.with_limit(Resource::TaskStateBytes, admitted.retained_bytes() - 1);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &metadata, &values, &too_small);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(matches!(
        result,
        Err(PlanAdmissionError::ResourceExhausted(error))
            if error.resource == Resource::TaskStateBytes
    ));
    assert_eq!(allocations, 0, "rejection must precede producer allocation");

    let metadata_too_small = limits.with_limit(
        Resource::MetadataAllocatedBytes,
        admitted.metadata_bytes() - 1,
    );
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &metadata, &values, &metadata_too_small);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(matches!(
        result,
        Err(PlanAdmissionError::ResourceExhausted(error))
            if error.resource == Resource::MetadataAllocatedBytes
    ));
    assert_eq!(allocations, 0, "metadata rejection must precede allocation");

    for resource in [
        Resource::PlanNodes,
        Resource::PlanDepth,
        Resource::SchemaNodes,
        Resource::SchemaDepth,
        Resource::WorkUnits,
        Resource::PlanEncodedBytes,
        Resource::TaskStateBytes,
    ] {
        let zero = limits.with_limit(resource, 0);
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let result = AdmittedPlan::try_i64_values("value", &[], &values, &zero);
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert!(matches!(
            result,
            Err(PlanAdmissionError::ResourceExhausted(error)) if error.resource == resource
        ));
        assert_eq!(allocations, 0, "{resource:?} rejection must not allocate");
    }

    let duplicate = [
        PlanMetadataEntry::new("id", &number),
        PlanMetadataEntry::new("id", &boolean),
    ];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &duplicate, &values, &limits);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(result.unwrap_err(), PlanAdmissionError::DuplicateMetadata);
    assert_eq!(allocations, 0, "duplicate rejection must not allocate");

    let other = MetadataValue::Other(serde_json::json!({"nested": [1, 2, 3]}));
    let unsupported = [PlanMetadataEntry::new("other", &other)];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &unsupported, &values, &limits);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(result.unwrap_err(), PlanAdmissionError::UnsupportedMetadata);
    assert_eq!(allocations, 0, "unsupported metadata must not allocate");

    let metadata_spec = MetadataValue::String("row_index".into());
    let reserved = [PlanMetadataEntry::new("delta.metadataSpec", &metadata_spec)];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &reserved, &values, &limits);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(result.unwrap_err(), PlanAdmissionError::UnsupportedMetadata);
    assert_eq!(
        allocations, 0,
        "metadata-column indexes require a separate admitted producer"
    );

    let encoded_limit = limits.with_limit(
        Resource::PlanEncodedBytes,
        admitted.shape().encoded_bytes() - 1,
    );
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_i64_values("value", &metadata, &values, &encoded_limit);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(matches!(
        result,
        Err(PlanAdmissionError::ResourceExhausted(error))
            if error.resource == Resource::PlanEncodedBytes
    ));
    assert_eq!(allocations, 0, "encoded-size rejection must not allocate");

    let shape = admitted.shape();
    let encoded = delta_kernel::plans::Operation::QueryPlan(admitted.into_plan()).to_proto_bytes();
    assert!(shape.encoded_bytes() >= encoded.len());
}

#[test]
fn admitted_string_values_preflight_variable_width_storage() {
    let values = ["", "nul\0snow 雪", "tail"];
    let limits = TaskLimits::qualification();
    ALLOCATED_BYTES.with(|count| count.set(Some(0)));
    let admitted = AdmittedPlan::try_string_values("text", &[], &values, &limits).unwrap();
    let allocated_bytes = ALLOCATED_BYTES.with(|count| count.replace(None).unwrap());

    assert!(allocated_bytes <= admitted.retained_bytes());
    let retained_bytes = admitted.retained_bytes();
    let shape = admitted.shape();
    let too_small = limits.with_limit(Resource::TaskStateBytes, retained_bytes - 1);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_string_values("text", &[], &values, &too_small);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(matches!(
        result,
        Err(PlanAdmissionError::ResourceExhausted(error))
            if error.resource == Resource::TaskStateBytes
    ));
    assert_eq!(allocations, 0, "rejection must precede producer allocation");

    let encoded_limit = limits.with_limit(Resource::PlanEncodedBytes, shape.encoded_bytes() - 1);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = AdmittedPlan::try_string_values("text", &[], &values, &encoded_limit);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(matches!(
        result,
        Err(PlanAdmissionError::ResourceExhausted(error))
            if error.resource == Resource::PlanEncodedBytes
    ));
    assert_eq!(allocations, 0, "encoded rejection must not allocate");

    let plan = admitted.into_plan();
    let Operator::Values(values) = &plan.nodes[0].op else {
        panic!("expected Values source")
    };
    assert_eq!(values.schema.fields().len(), 1);
    let field = values.schema.fields().next().unwrap();
    assert_eq!(field.data_type(), &DataType::STRING);
    assert!(!field.is_nullable());
    assert_eq!(values.rows[1][0], Scalar::String("nul\0snow 雪".into()));
    let encoded = delta_kernel::plans::Operation::QueryPlan(plan).to_proto_bytes();
    assert!(shape.encoded_bytes() >= encoded.len());
}
