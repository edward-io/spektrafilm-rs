//! Update the saving-space label while retaining unrelated XMP namespaces.
use anyhow::{Result, bail};
use quick_xml::{
    NsReader, Writer,
    events::{BytesStart, Event},
    name::ResolveResult,
};
const PHOTOSHOP: &[u8] = b"http://ns.adobe.com/photoshop/1.0/";
const RDF: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#";
fn description_name(name: &str) -> String {
    name.rsplit_once(':')
        .map(|(prefix, _)| format!("{prefix}:Description"))
        .unwrap_or_else(|| "Description".into())
}
fn matches(ns: ResolveResult<'_>, local: &[u8], wanted_ns: &[u8], wanted_name: &[u8]) -> bool {
    local == wanted_name && matches!(ns,ResolveResult::Bound(v) if v.as_ref()==wanted_ns)
}
pub(super) fn saving_profile(source: Option<&[u8]>, space: &str, encoded: bool) -> Result<Vec<u8>> {
    let source=source.unwrap_or(br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>"#);
    let text = std::str::from_utf8(source)?;
    let mut prefix = "sf_icc".to_owned();
    while text.contains(&prefix) {
        prefix.push('_');
    }
    let value = if encoded {
        space.to_owned()
    } else {
        format!("{space} (linear)")
    };
    let ns_attribute = format!("xmlns:{prefix}");
    let value_attribute = format!("{prefix}:ICCProfile");
    let mut reader = NsReader::from_reader(source);
    let mut writer = Writer::new(Vec::with_capacity(source.len() + 128));
    let mut skipped = 0usize;
    let mut inserted = false;
    loop {
        let event = reader.read_event()?;
        if skipped > 0 {
            match event {
                Event::Start(_) => skipped += 1,
                Event::End(_) => skipped -= 1,
                Event::Eof => bail!("unterminated XMP profile field"),
                _ => {}
            }
            continue;
        }
        match event {
            Event::Start(ref start) | Event::Empty(ref start) => {
                let (ns, local) = reader.resolve_element(start.name());
                if matches(ns, local.as_ref(), PHOTOSHOP, b"ICCProfile") {
                    if matches!(event, Event::Start(_)) {
                        skipped = 1;
                    }
                    continue;
                }
                let (ns, local) = reader.resolve_element(start.name());
                let description = matches(ns, local.as_ref(), RDF, b"Description");
                let (ns, local) = reader.resolve_element(start.name());
                let empty_rdf =
                    matches!(event, Event::Empty(_)) && matches(ns, local.as_ref(), RDF, b"RDF");
                let name = std::str::from_utf8(start.name().as_ref())?.to_owned();
                let mut clean = BytesStart::new(name);
                for attribute in start.attributes() {
                    let attribute = attribute?;
                    let (ns, local) = reader.resolve_attribute(attribute.key);
                    if !matches(ns, local.as_ref(), PHOTOSHOP, b"ICCProfile") {
                        clean.push_attribute(attribute);
                    }
                }
                if description && !inserted {
                    clean.push_attribute((ns_attribute.as_str(), std::str::from_utf8(PHOTOSHOP)?));
                    clean.push_attribute((value_attribute.as_str(), value.as_str()));
                    inserted = true;
                }
                if empty_rdf && !inserted {
                    let name = description_name(std::str::from_utf8(start.name().as_ref())?);
                    let end = clean.to_end().into_owned();
                    writer.write_event(Event::Start(clean))?;
                    let mut child = BytesStart::new(name);
                    child.push_attribute((ns_attribute.as_str(), std::str::from_utf8(PHOTOSHOP)?));
                    child.push_attribute((value_attribute.as_str(), value.as_str()));
                    writer.write_event(Event::Empty(child))?;
                    writer.write_event(Event::End(end))?;
                    inserted = true;
                    continue;
                }
                writer.write_event(if matches!(event, Event::Start(_)) {
                    Event::Start(clean)
                } else {
                    Event::Empty(clean)
                })?;
            }
            Event::End(ref end) => {
                let (ns, local) = reader.resolve_element(end.name());
                if matches(ns, local.as_ref(), RDF, b"RDF") && !inserted {
                    let name = description_name(std::str::from_utf8(end.name().as_ref())?);
                    let mut description = BytesStart::new(name);
                    description
                        .push_attribute((ns_attribute.as_str(), std::str::from_utf8(PHOTOSHOP)?));
                    description.push_attribute((value_attribute.as_str(), value.as_str()));
                    writer.write_event(Event::Empty(description))?;
                    inserted = true;
                }
                writer.write_event(event)?;
            }
            Event::Eof => break,
            _ => writer.write_event(event)?,
        }
    }
    if !inserted {
        bail!("XMP packet has no RDF container");
    }
    Ok(writer.into_inner())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saving_profile_replaces_attribute_and_element_aliases() {
        let source=br#"<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><r:Description xmlns:p="http://ns.adobe.com/photoshop/1.0/" p:ICCProfile="old" xmlns:dc="http://purl.org/dc/elements/1.1/" dc:title="Keep &amp; preserve"><p:ICCProfile>also old</p:ICCProfile></r:Description></r:RDF>"#;
        let output = saving_profile(Some(source), "Display P3", false).unwrap();
        let text = String::from_utf8(output.clone()).unwrap();
        assert!(!text.contains("old"));
        assert!(text.contains("Display P3 (linear)"));
        assert!(text.contains("Keep &amp; preserve"));
        let output = saving_profile(Some(&output), "sRGB", true).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(!text.contains("Display P3"));
        assert!(text.contains("sRGB"));
        let empty = saving_profile(
            Some(br#"<RDF xmlns="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>"#),
            "sRGB",
            true,
        )
        .unwrap();
        assert!(String::from_utf8(empty).unwrap().contains("sRGB"));
        let fresh = saving_profile(None, "ProPhoto RGB", true).unwrap();
        assert!(String::from_utf8(fresh).unwrap().contains("ProPhoto RGB"));
    }
}
