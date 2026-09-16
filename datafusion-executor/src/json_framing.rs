//! Allocation-free framing before Arrow's tape decoder sees admitted whole-log bytes.

use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

/// Lexical work and framing observations; no Delta action interpretation is performed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonFraming {
    pub records: usize,
    pub tokens: usize,
    pub max_record_bytes: usize,
    pub max_depth: usize,
}

/// Validates complete object documents and counts unknown fields as well as selected fields.
pub(crate) fn preflight(bytes: &[u8], limits: TaskLimits) -> Result<JsonFraming, OperationFailure> {
    check(Resource::InputBytes, bytes.len(), limits)?;
    check(Resource::WorkUnits, bytes.len(), limits)?;
    let mut parser = Parser {
        bytes,
        pos: 0,
        limits,
        framing: JsonFraming::default(),
    };
    parser.space();
    while parser.pos < bytes.len() {
        let start = parser.pos;
        if parser.peek() != Some(b'{') {
            return Err(OperationFailure::malformed_response());
        }
        parser.value(0)?;
        parser.framing.records += 1;
        check(Resource::Records, parser.framing.records, limits)?;
        let size = parser.pos - start;
        check(Resource::PartialJsonBytes, size, limits)?;
        parser.framing.max_record_bytes = parser.framing.max_record_bytes.max(size);
        parser.space();
    }
    Ok(parser.framing)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: TaskLimits,
    framing: JsonFraming,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn space(&mut self) {
        while self
            .peek()
            .is_some_and(|b| matches!(b, b' ' | b'\r' | b'\n' | b'\t'))
        {
            self.pos += 1;
        }
    }
    fn take(&mut self, byte: u8) -> Result<(), OperationFailure> {
        if self.peek() != Some(byte) {
            return Err(OperationFailure::malformed_response());
        }
        self.pos += 1;
        Ok(())
    }
    fn token(&mut self) -> Result<(), OperationFailure> {
        self.framing.tokens += 1;
        check(Resource::WorkUnits, self.framing.tokens, self.limits)
    }
    fn value(&mut self, depth: usize) -> Result<(), OperationFailure> {
        self.token()?;
        match self.peek() {
            Some(b'{') | Some(b'[') => {
                let object = self.peek() == Some(b'{');
                let close = if object { b'}' } else { b']' };
                let depth = depth + 1;
                // The finite qualification ceiling also bounds this host's stack independently
                // of caller-provided larger allowances.
                let limit = self
                    .limits
                    .limit(Resource::SchemaDepth)
                    .min(TaskLimits::qualification().limit(Resource::SchemaDepth));
                if depth > limit {
                    return Err(ResourceExhausted {
                        resource: Resource::SchemaDepth,
                        limit,
                        observed: depth,
                    }
                    .into());
                }
                self.framing.max_depth = self.framing.max_depth.max(depth);
                self.pos += 1;
                self.space();
                if self.peek() == Some(close) {
                    self.pos += 1;
                    return self.token();
                }
                loop {
                    if object {
                        self.token()?;
                        self.string()?;
                        self.space();
                        self.take(b':')?;
                        self.space();
                    }
                    self.value(depth)?;
                    self.space();
                    if self.peek() == Some(close) {
                        self.pos += 1;
                        return self.token();
                    }
                    self.take(b',')?;
                    self.space();
                }
            }
            Some(b'"') => self.string(),
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(OperationFailure::malformed_response()),
        }
    }
    fn string(&mut self) -> Result<(), OperationFailure> {
        self.take(b'"')?;
        let start = self.pos;
        loop {
            match self.peek() {
                Some(b'"') => {
                    // UTF-8 validation borrows input and allocates nothing.
                    if std::str::from_utf8(&self.bytes[start..self.pos]).is_err() {
                        return Err(OperationFailure::malformed_response());
                    }
                    self.pos += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {
                            self.pos += 1
                        }
                        Some(b'u') => {
                            self.pos += 1;
                            for _ in 0..4 {
                                if !self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                                    return Err(OperationFailure::malformed_response());
                                }
                                self.pos += 1;
                            }
                            // Arrow validates surrogate pairing.
                        }
                        _ => return Err(OperationFailure::malformed_response()),
                    }
                }
                Some(0..=31) | None => return Err(OperationFailure::malformed_response()),
                _ => self.pos += 1,
            }
        }
    }
    fn literal(&mut self, value: &[u8]) -> Result<(), OperationFailure> {
        for &byte in value {
            self.take(byte)?;
        }
        Ok(())
    }
    fn digits(&mut self) -> Result<(), OperationFailure> {
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(OperationFailure::malformed_response());
        }
        Ok(())
    }
    fn number(&mut self) -> Result<(), OperationFailure> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        if self.peek() == Some(b'0') {
            self.pos += 1;
        } else {
            self.digits()?;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.digits()?;
        }
        Ok(())
    }
}

fn check(resource: Resource, observed: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
    let limit = limits.limit(resource);
    if observed > limit {
        return Err(ResourceExhausted {
            resource,
            limit,
            observed,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use delta_kernel::tasks::FailureKind;

    use super::*;

    #[test]
    fn unknown_fields_and_multiple_documents_consume_budget() {
        let input = br#"{"unknown":[1,true,null,{"nested":"\\\""}]} {}"#;
        let result = preflight(input, TaskLimits::qualification()).unwrap();
        assert_eq!(result.records, 2);
        assert_eq!(result.max_depth, 3);
        assert!(result.tokens > 10);
        let failure = preflight(
            input,
            TaskLimits::qualification().with_limit(Resource::SchemaDepth, 2),
        )
        .unwrap_err();
        assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::SchemaDepth));
    }

    #[test]
    fn rejects_incomplete_or_malformed_framing() {
        for input in [
            "[]",
            "{",
            "{]",
            "{\"x\":01}",
            "{\"x\":1.}",
            "{\"x\":tru}",
            "{\"x\":\"\\q\"}",
            "{\"x\":[1,]}",
            "{} trailing",
        ] {
            assert!(
                preflight(input.as_bytes(), TaskLimits::qualification()).is_err(),
                "{input}"
            );
        }
    }
}
