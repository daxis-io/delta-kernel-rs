//! Temporary owners of url 2.5.8 join, including rejected authority inputs.
//! The ordinary parser remains authoritative for normalization and confinement.
use super::json_schema_shape::vector_peak;

pub(super) fn join_peak(root_bytes: usize, path: &str) -> Option<usize> {
    let bytes = path.len();
    // This sufficient (not necessary) classification reaches parse_relative's
    // path branch without scheme, authority, file-host or IDNA parsing. Query
    // and fragment parsing use the same serialization String. Each input byte
    // becomes at most three percent-encoding bytes; dot segments only remove.
    let relative = !path.starts_with('/')
        && path
            .bytes()
            .all(|b| b != b':' && b != b'\\' && b > 0x20 && b != 0x7f);
    let path_serialization = root_bytes.checked_add(bytes.checked_mul(3)?)?;
    if relative {
        return vector_peak::<u8>(path_serialization);
    }
    general_join_peak(root_bytes, bytes)
}

/// Semantic byte/scalar work of join plus the caller's path clone and
/// confinement comparison. Callers prepay path.len before this classification.
pub(super) fn join_work(root_bytes: usize, path: &str) -> Option<usize> {
    let bytes = path.len();
    let relative = !path.starts_with('/')
        && path
            .bytes()
            .all(|b| b != b':' && b != b'\\' && b > 0x20 && b != 0x7f);
    if relative {
        // Clone path, owner-bound classification, parser scheme scan, path
        // input, and at most three percent-encoded output bytes per byte.
        // Base copy, shorten_path scan, and final confinement prefix compare.
        return bytes
            .checked_mul(1 + 1 + 1 + 1 + 3)?
            .checked_add(root_bytes.checked_mul(3)?);
    }
    // Same selected ICU decomposition bound as general_join_peak: at most
    // eighteen scalars per input byte, already recursively flattened.
    let chars = bytes.checked_mul(18)?;
    let ascii = chars
        .checked_mul(10)?
        .checked_add(chars.checked_add(1)?.checked_mul(5)?)?;
    let serialized = root_bytes
        .checked_add(bytes.checked_mul(3)?)?
        .checked_add(ascii)?;
    // URL parser branches are mutually exclusive, but charge every reached
    // byte walk: scheme, slash/authority, host/port split, whitespace filter,
    // percent decode, IPv4 parts/numbers, path, query, fragment, serialization,
    // original path clone, base copy and final confinement comparison.
    const URL_WALKS: &[&str] = &[
        "scheme",
        "authority",
        "host/port",
        "whitespace",
        "percent decode",
        "IPv4 parts",
        "IPv4 numbers",
        "path",
        "query",
        "fragment",
        "serialization",
        "path clone",
        "base copy",
        "confinement",
    ];
    // idna1.1 uts46: ASCII prefix/label scanning, map/normalize, ACE decode,
    // normalized comparison, validity, hyphens, bidi, joiners, label output,
    // error-policy label/output walks. Count alternative output paths as well.
    const LABEL_WALKS: &[&str] = &[
        "prefix",
        "labels",
        "mapping",
        "normalization",
        "ACE",
        "normalized equality",
        "validity",
        "hyphens",
        "bidi",
        "joiners",
        "label output",
        "error labels",
        "error output",
    ];
    let linear = serialized
        .checked_mul(URL_WALKS.len())?
        .checked_add(chars.checked_mul(LABEL_WALKS.len())?)?;
    // Punycode encode: <=C outer passes, each minimum search + input scan.
    // ICU normalization: combining-class ordering and canonical composition
    // each bounded by all scalar pairs, for mapping and ACE revalidation.
    let scalar_pairs = chars.checked_mul(chars)?.checked_mul(2 + 2 + 2)?;
    // Punycode decode updates all prior insertion indices, then stable sorts
    // them. <=B decoded insertions; each operation is bounded by all pairs.
    let decode_pairs = bytes.checked_mul(bytes)?.checked_mul(1 + 1)?;
    // Encode q loses a factor >=10 per emitted digit (<=10 digits for u32).
    // adapt repeatedly divides a u32 by BASE-T_MIN>=2: <=u32::BITS steps.
    let digits = chars.checked_mul(10 + u32::BITS as usize)?;
    linear
        .checked_add(scalar_pairs)?
        .checked_add(decode_pairs)?
        .checked_add(digits)
}

fn smallvec_peak<T>(items: usize, inline: usize) -> Option<usize> {
    // SmallVec reserve rounds to a power of two. On spilling, capacity is at
    // most twice max(requested, inline); retain old and new simultaneously.
    vector_peak::<T>(items.max(inline))
}

fn general_join_peak(root_bytes: usize, bytes: usize) -> Option<usize> {
    // Both selected ICU normalizers (2.1.1 and 2.2.0) use a four-bit
    // decomposition length +2, or the special FDFA decomposition of 18 chars.
    // Decompositions are already flattened. Composition can only reduce this.
    let chars = bytes.checked_mul(18)?;
    let labels = chars.checked_add(1)?;
    // Punycode's u32 q is divided by BASE-t >=10 on each emitted digit:
    // at most ten digits per scalar, plus '-' and the four-byte xn-- prefix
    // per label. ASCII/opaque hosts fit this bound too. Count empty labels.
    let ascii = chars.checked_mul(10)?.checked_add(labels.checked_mul(5)?)?;
    let serialization = root_bytes
        .checked_add(bytes.checked_mul(3)?)?
        .checked_add(ascii)?
        .checked_add(2 + 8 * 4 + 7 + 1 + 5)?;
    // Parser::parse_host_and_port can collect a whitespace-filtered host;
    // percent_decode may independently allocate its byte owner. The IDNA sink
    // remains live while the parser appends it to URL serialization.
    let text = vector_peak::<u8>(serialization)?
        .checked_add(vector_peak::<u8>(bytes)?.checked_mul(2)?)?
        .checked_add(vector_peak::<u8>(ascii)?)?;
    // IDNA domain chars, label-kind enum (tag + borrowed byte slice), decoded
    // label chars and PunycodeDecoder's insertion list. The two label buffers
    // occur on alternative paths; counting both is a sum of reached sites.
    let idna = smallvec_peak::<char>(chars, 253)?
        .checked_add(smallvec_peak::<[usize; 3]>(labels, 8)?)?
        .checked_add(smallvec_peak::<char>(bytes, 59)?.checked_mul(2)?)?
        // PunycodeDecoder sorts its insertion list stably. The four-times
        // capacity term also bounds current <2n backing plus <=n sort scratch;
        // the reserve relocation and stable sort occur at separate sites.
        .checked_add(smallvec_peak::<(usize, char)>(bytes, 59)?)?;
    // ICU CharacterAndClass is one u32, inline17. Its stable sort can request
    // a full-length scratch buffer; include it independently of the growable
    // normalization buffer. Immutable compiled Unicode data is borrowed.
    let normalization = smallvec_peak::<u32>(chars, 17)?.checked_add(vector_peak::<u32>(chars)?)?;
    // IPv4 parser collects ALL dot-separated parts before its <=4 validation;
    // only then can it grow the four-number vector. Include invalid inputs.
    let ipv4 = vector_peak::<&str>(ascii.checked_add(1)?)?.checked_add(vector_peak::<u32>(4)?)?;
    [text, idna, normalization, ipv4]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_work_includes_long_roots_paths_and_authority_pairs() {
        let short = join_work(20, "a.parquet").unwrap();
        assert_eq!(join_work(1020, "a.parquet").unwrap() - short, 3 * 1000);
        assert_eq!(join_work(20, "long-a.parquet").unwrap() - short, 7 * 5);
        assert!(join_work(20, "//xn--r8jz45g.xn--zckzah/a").unwrap() > short);
        assert!(join_work(20, "https://例え.テスト/a").unwrap() > short);
        assert_eq!(join_work(usize::MAX, "a.parquet"), None);
        assert_eq!(join_work(usize::MAX, "//a.parquet"), None);
    }
    #[test]
    fn join_serialization_and_overflow_checks_cover_authority_forms() {
        for root in [
            "https://example.com/table/",
            "file:///C:/table/",
            "memory:///table/",
        ] {
            let root = url::Url::parse(root).unwrap();
            for path in [
                "a.parquet",
                "dir/a%20b.parquet",
                "../x",
                "%2e%2e/x",
                "//例え.テスト/a",
                "https://xn--r8jz45g.xn--zckzah/a",
                "\\\\example.com/a",
                "http://0.0.0.0.0/a",
                " https://[::1]:65535/a?x#y",
                "a?x#y",
                "",
            ] {
                let peak = join_peak(root.as_str().len(), path).unwrap();
                if let Ok(joined) = root.join(path) {
                    assert!(joined.as_str().len() <= peak, "{root} + {path}");
                }
            }
        }
        assert_eq!(join_peak(usize::MAX, "a"), None);
        assert_eq!(general_join_peak(0, usize::MAX), None);
        assert!(join_peak(30, "a.parquet").unwrap() < join_peak(30, "//a.parquet").unwrap());
    }
}
