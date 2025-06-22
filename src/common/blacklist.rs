use std::fs;
use std::io::{self, BufRead};

#[derive(Clone)]
pub struct Blacklist {
    addresses: Vec<String>,
}

impl Blacklist {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let mut addresses = Vec::new();

        let file = fs::File::open(file_path)?;
        let reader = io::BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            addresses.push(line);
        }

        Ok(Blacklist { addresses })
    }

    pub fn empty() -> Self {
        let addresses = Vec::<String>::new();
        Blacklist { addresses }
    }

    pub fn length(self) -> usize {
        self.addresses.len()
    }

    // Check if an address is in the blacklist
    pub fn is_blacklisted(&self, address: &str) -> bool {
        self.addresses.contains(&address.to_string())
    }
}
