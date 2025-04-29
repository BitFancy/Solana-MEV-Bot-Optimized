use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom, Read};
use serde_json::Value;

pub fn print_json_segment(file_path: &str, start: u64, length: usize) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(start))?;

    let mut buffer = vec![0; length];
    reader.read_exact(&mut buffer)?;

    let segment = String::from_utf8_lossy(&buffer);
    println!("{}", segment);

    Ok(())
}