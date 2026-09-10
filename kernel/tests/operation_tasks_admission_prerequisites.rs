#![cfg(feature = "operation-tasks")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel::expressions::ColumnName;
use delta_kernel::plans::ir::nodes::{DynamicScan, FileType};
use delta_kernel::schema::{DataType, StructField, StructType, ToSchema};
use delta_kernel::FileMeta;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation();
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
fn plan_shape_bounds_nested_scalars_schemas_metadata_and_all_operators() {
    use delta_kernel::expressions::{ArrayData, DecimalData, MapData, Scalar, StructData};
    use delta_kernel::plans::ir::nodes::{
        Agg, Aggregate, Filter, NonNullByOperands, ScanFile, ScanJson, ScanParquet, SemiJoin,
        UnionAll, Values,
    };
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::schema::{ArrayType, DecimalType, MapType, MetadataValue};

    let field = StructField::nullable("text\0\n雪", DataType::STRING).with_metadata([
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
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    for expr in expression_wrappers(Expression::Column(ColumnName::new(["field", "nested"]))) {
        assert_plan_shape_bound(expression_plan(expr));
    }
    for unknown in [
        Expression::Unknown("hidden".into()),
        Expression::Predicate(Box::new(Predicate::Unknown("hidden".into()))),
    ] {
        for expr in expression_wrappers(unknown) {
            let plan = expression_plan(expr);
            ALLOCATIONS.with(|count| count.set(Some(0)));
            let result = check_plan_shape(&plan, &TaskLimits::qualification());
            let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
            assert_eq!(result, Err(PlanShapeError::UnsupportedExpression));
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
    assert!(
        matches!(check_plan_shape(&sparse, &limits.with_limit(Resource::WorkUnits, 100)), Err(PlanShapeError::ResourceExhausted(e)) if e.resource == Resource::WorkUnits)
    );
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
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};

    let op = Arc::new(NeverInvoke);
    let expr = Expression::Opaque(OpaqueExpression {
        op: op.clone(),
        exprs: vec![],
    });
    for expr in expression_wrappers(expr) {
        let plan = expression_plan(expr);
        let before = Arc::strong_count(&op);
        assert_eq!(
            check_plan_shape(&plan, &TaskLimits::qualification()),
            Err(PlanShapeError::UnsupportedExpression)
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
    assert_plan_shape_bound(Plan {
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
    use delta_kernel::tasks::{PlanShapeError, TaskLimits};
    let op = Arc::new(NeverInvoke);
    let expr = Expression::Predicate(Box::new(Predicate::Opaque(OpaquePredicate {
        op: op.clone(),
        exprs: vec![],
    })));
    for expr in expression_wrappers(expr) {
        assert_eq!(
            check_plan_shape(&expression_plan(expr), &TaskLimits::qualification()),
            Err(PlanShapeError::UnsupportedExpression)
        );
    }
    assert_eq!(Arc::strong_count(&op), 1);
}
