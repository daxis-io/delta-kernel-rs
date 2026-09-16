#![cfg(feature = "nanosecond-timestamps")]

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::actions::{Metadata, Protocol};
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{DataType, PrimitiveType, StructField, StructType};
use delta_kernel::table_configuration::TableConfiguration;
use delta_kernel::table_features::TableFeature;
use rstest::rstest;

#[rstest]
#[case("1970-01-01T00:00:00.123456789Z", 123456789)]
#[case("1969-12-31T23:59:59.999999999Z", -1)]
#[case("1970-01-01 00:00:00", 0)]
#[case("2262-04-11T23:47:16.854775807Z", i64::MAX)]
#[case("1970-01-01T00:00:00+05:00", -18_000_000_000_000)]
#[case("1970-01-01T00:00:00-05:00", 18_000_000_000_000)]
#[case("1970-01-01T00:00:00+05:30", -19_800_000_000_000)]
#[case("1970-01-01T00:00:00-03:30", 12_600_000_000_000)]
#[case("1970-01-01T00:00:00+0530", -19_800_000_000_000)]
#[case("1970-01-01T00:00:00.123456789+00:01", -59_876_543_211)]
#[case("1969-12-31T23:59:59.999999999-00:01", 59_999_999_999)]
fn nanos_parsing_and_partition_serialization_preserve_precision(
    #[case] text: &str,
    #[case] value: i64,
) {
    let scalar = PrimitiveType::TimestampNanos.parse_scalar(text).unwrap();
    assert_eq!(scalar, Scalar::TimestampNanos(value));
    let serialized = delta_kernel::partition::serialization::serialize_partition_value(&scalar)
        .unwrap()
        .unwrap();
    assert_eq!(
        PrimitiveType::TimestampNanos
            .parse_scalar(&serialized)
            .unwrap(),
        scalar
    );
    assert!(serialized.ends_with('Z'));
    assert_eq!(serialized.rsplit_once('.').unwrap().1.len(), 10);
}

#[rstest]
#[case("2262-04-11T23:47:16.854775808Z")]
#[case("1677-09-21T00:12:43.145224191Z")]
#[case("2262-04-11T23:47:16.854775807-00:01")]
#[case("1677-09-21T00:12:43.145224192+00:01")]
#[case("not a timestamp")]
fn nanos_parsing_rejects_overflow_and_malformed_values(#[case] text: &str) {
    assert!(PrimitiveType::TimestampNanos.parse_scalar(text).is_err());
}

#[rstest]
fn nanos_schema_requires_protocol_feature(
    #[values(false, true)] nested: bool,
    #[values(false, true)] feature: bool,
) {
    let leaf =
        StructType::try_new([StructField::nullable("ts", DataType::TIMESTAMP_NANOS)]).unwrap();
    let schema = if nested {
        StructType::try_new([StructField::nullable("nested", leaf)]).unwrap()
    } else {
        leaf
    };
    let metadata =
        Metadata::try_new(None, None, Arc::new(schema), vec![], 0, HashMap::new()).unwrap();
    let features = if feature {
        vec![TableFeature::TimestampNanos]
    } else {
        vec![]
    };
    let protocol: Protocol = serde_json::from_value(serde_json::json!({"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":features,"writerFeatures":features})).unwrap();
    let result = TableConfiguration::try_new(
        metadata,
        protocol,
        url::Url::parse("memory:///").unwrap(),
        0,
    );
    assert_eq!(result.is_ok(), feature);
}

#[cfg(feature = "declarative-plans")]
#[test]
fn nanos_wire_identifiers_preserve_existing_fields_and_exact_value() {
    use delta_kernel::plans::proto::{expressions, schema};
    assert_eq!(schema::SimplePrimitiveType::Timestamp as i32, 11);
    assert_eq!(schema::SimplePrimitiveType::TimestampNtz as i32, 12);
    assert_eq!(schema::SimplePrimitiveType::TimestampNanos as i32, 16);
    let wire = schema::PrimitiveType::from(&PrimitiveType::TimestampNanos);
    assert_eq!(
        PrimitiveType::try_from(wire).unwrap(),
        PrimitiveType::TimestampNanos
    );
    let wire = expressions::Scalar::from(&Scalar::TimestampNanos(-1));
    assert!(matches!(
        wire.value,
        Some(expressions::scalar::Value::TimestampNanos(-1))
    ));
}

#[cfg(feature = "default-engine-base")]
#[test]
fn nanos_arrow_parquet_roundtrip_keeps_signed_values_and_statistics() {
    use delta_kernel::arrow::array::{Array, TimestampNanosecondArray};
    use delta_kernel::arrow::datatypes::{DataType as ArrowType, Field, Schema, TimeUnit};
    use delta_kernel::arrow::record_batch::RecordBatch;
    use delta_kernel::engine::arrow_conversion::{TryIntoArrow, TryIntoKernel};
    use delta_kernel::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use delta_kernel::parquet::arrow::ArrowWriter;
    use delta_kernel::parquet::file::statistics::Statistics;

    let arrow_type: ArrowType = (&DataType::TIMESTAMP_NANOS).try_into_arrow().unwrap();
    assert_eq!(
        arrow_type,
        ArrowType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
    );
    let roundtrip: DataType = (&arrow_type).try_into_kernel().unwrap();
    assert_eq!(roundtrip, DataType::TIMESTAMP_NANOS);
    let values = vec![Some(-1), Some(0), None, Some(123456789), Some(i64::MAX)];
    let array = TimestampNanosecondArray::from(values.clone()).with_timezone("UTC");
    let schema = Arc::new(Schema::new(vec![Field::new("ts", arrow_type, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let scalar_array = Scalar::TimestampNanos(-1).to_array(2).unwrap();
    let scalar_array = scalar_array
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(scalar_array.values(), &[-1, -1]);
    assert_eq!(
        Scalar::Null(DataType::TIMESTAMP_NANOS)
            .to_array(2)
            .unwrap()
            .null_count(),
        2
    );

    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes)).unwrap();
    let stats = reader
        .metadata()
        .row_group(0)
        .column(0)
        .statistics()
        .unwrap();
    match stats {
        Statistics::Int64(stats) => {
            assert_eq!(stats.min_opt(), Some(&-1));
            assert_eq!(stats.max_opt(), Some(&i64::MAX));
        }
        other => panic!("unexpected nanos statistics: {other:?}"),
    }
    let mut reader = reader.build().unwrap();
    let decoded = reader.next().unwrap().unwrap();
    let actual = decoded
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(actual.iter().collect::<Vec<_>>(), values);
    assert!(reader.next().is_none());
}

#[cfg(feature = "operation-tasks")]
#[test]
fn nanos_plan_shape_accounts_wire_and_rejects_tight_budget() {
    use delta_kernel::plans::ir::nodes::Values;
    use delta_kernel::plans::ir::plan::{Plan, PlanNode};
    use delta_kernel::tasks::{PlanShape, Resource, TaskLimits};
    use prost::Message;
    let schema = Arc::new(
        StructType::try_new([StructField::not_null("ts", DataType::TIMESTAMP_NANOS)]).unwrap(),
    );
    let plan = Plan {
        nodes: vec![PlanNode::new(
            Values::new(schema, vec![vec![Scalar::TimestampNanos(-1)]]),
            vec![],
        )],
    };
    let limits = TaskLimits::qualification();
    let shape = PlanShape::check_literals(&plan, &limits, &mut [0]).unwrap();
    let wire = delta_kernel::plans::proto::plan::Plan::from(&plan);
    assert!(shape.encoded_bytes() >= wire.encoded_len());
    let tight = limits.with_limit(Resource::PlanEncodedBytes, shape.encoded_bytes() - 1);
    assert!(PlanShape::check_literals(&plan, &tight, &mut [0]).is_err());
}

#[rstest]
fn nanos_protocol_requires_both_reader_and_writer_features(
    #[values(false, true)] reader: bool,
    #[values(false, true)] writer: bool,
) {
    let schema = Arc::new(
        StructType::try_new([StructField::nullable("ts", DataType::TIMESTAMP_NANOS)]).unwrap(),
    );
    let metadata = Metadata::try_new(None, None, schema, vec![], 0, HashMap::new()).unwrap();
    let protocol = serde_json::from_value::<Protocol>(serde_json::json!({
        "minReaderVersion":3,"minWriterVersion":7,
        "readerFeatures":if reader {vec!["timestampNanos"]} else {vec![]},
        "writerFeatures":if writer {vec!["timestampNanos"]} else {vec![]}
    }));
    if reader != writer {
        assert!(
            protocol.is_err(),
            "asymmetric reader/writer features must be rejected during protocol decoding"
        );
        return;
    }
    let protocol = protocol.unwrap();
    assert_eq!(
        TableConfiguration::try_new(
            metadata,
            protocol,
            url::Url::parse("memory:///").unwrap(),
            0
        )
        .is_ok(),
        reader && writer
    );
}

#[rstest]
fn nanos_native_creation_and_snapshot_roundtrip(
    #[values(0, 1, 2, 3)] nesting: u8,
) -> delta_kernel::DeltaResult<()> {
    use delta_kernel::committer::FileSystemCommitter;
    use delta_kernel::schema::{ArrayType, MapType};
    use delta_kernel::transaction::create_table::create_table;
    let leaf = DataType::TIMESTAMP_NANOS;
    let data_type = match nesting {
        0 => leaf,
        1 => StructType::try_new([StructField::nullable("nested", leaf)])?.into(),
        2 => ArrayType::new(leaf, true).into(),
        3 => MapType::new(DataType::STRING, leaf, true).into(),
        _ => unreachable!(),
    };
    let schema = Arc::new(StructType::try_new([StructField::nullable(
        "ts", data_type,
    )])?);
    let (_directory, path, engine) = test_utils::test_table_setup()?;
    create_table(&path, schema.clone(), "PhaseD/nanos")
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_committed();
    let snapshot = delta_kernel::Snapshot::builder_for(&path).build(engine.as_ref())?;
    assert_eq!(snapshot.version(), 0);
    assert_eq!(snapshot.schema().as_ref(), schema.as_ref());
    assert!(snapshot
        .table_configuration()
        .is_feature_supported(&TableFeature::TimestampNanos));
    Ok(())
}
