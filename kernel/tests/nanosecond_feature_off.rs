#![cfg(all(feature = "declarative-plans", not(feature = "nanosecond-timestamps")))]

#[test]
fn nanos_schema_wire_decode_is_typed_unsupported_when_disabled() {
    use delta_kernel::plans::proto::schema::{primitive_type, PrimitiveType, SimplePrimitiveType};
    let wire = PrimitiveType {
        kind: Some(primitive_type::Kind::Simple(
            SimplePrimitiveType::TimestampNanos as i32,
        )),
    };
    let error = delta_kernel::schema::PrimitiveType::try_from(wire).unwrap_err();
    assert!(matches!(error, delta_kernel::Error::Unsupported(_)));
    assert!(
        serde_json::from_str::<delta_kernel::schema::PrimitiveType>("\"timestamp_nanos\"").is_err()
    );
}
