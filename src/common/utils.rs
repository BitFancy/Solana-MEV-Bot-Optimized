use anyhow::Result;
use borsh::{BorshDeserialize, BorshSerialize};
use fern::colors::{Color, ColoredLevelConfig};
use log::{info, LevelFilter};
use solana_program::pubkey::Pubkey;
use solana_sdk::bs58;
use core::mem;
use std::{collections::HashMap, fs::{File, OpenOptions}};
use thiserror::Error;
use reqwest::Error;
use std::io::{BufWriter, Write};

use crate::{arbitrage::types::{SwapPathResult, TokenInArb, TokenInfos}, common::constants::{
    Env, PROJECT_NAME
}};
use solana_client::rpc_client::RpcClient;

// Function to format our console logs
pub fn setup_logger() -> Result<(), fern::InitError> {
    let colors = ColoredLevelConfig {
        trace: Color::Cyan,
        debug: Color::Magenta,
        info: Color::Green,
        warn: Color::Red,
        error: Color::BrightRed,
        ..ColoredLevelConfig::new()
    };

    let mut base_config = fern::Dispatch::new();

    //Console out
    let stdout_config = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{}[{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                colors.color(record.level()),
                message
            ))
        })
        .chain(std::io::stdout())
        .level(log::LevelFilter::Error)
        .level_for(PROJECT_NAME, LevelFilter::Info);
    
    //File logs
    let file_config = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{}{} [{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                chrono::Local::now().format("[%d/%m/%Y]"),
                record.level(),
                record.target(),
                message
            ))
        })
        .chain(fern::log_file("logs\\program.log")?)
        .level(log::LevelFilter::Error)
        .level_for(PROJECT_NAME, LevelFilter::Info);
    //Errors logs
    let errors_config = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{}{} [{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                chrono::Local::now().format("[%d/%m/%Y]"),
                record.level(),
                record.target(),
                message
            ))
        })
        .chain(fern::log_file("logs\\errors.log")?)
        .level(log::LevelFilter::Error);

    base_config
        .chain(file_config)
        .chain(errors_config)
        .chain(stdout_config)
        .apply()?;
    Ok(())
}

pub fn write_file_swap_path_result(path: String, content_raw: SwapPathResult) -> Result<()> {
    File::create(path.clone());

    let file = OpenOptions::new().read(true).write(true).open(path.clone())?;
    let mut writer = BufWriter::new(&file);

    writer.write_all(serde_json::to_string(&content_raw)?.as_bytes())?;
    writer.flush()?;
    info!("Data written to '{}' successfully.", path);

    Ok(())
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ParsePubkeyError {
    #[error("String is the wrong size")]
    WrongSize,
    #[error("Invalid Base58 string")]
    Invalid,
}

type Err = ParsePubkeyError;

/// Maximum string length of a base58 encoded pubkey
const MAX_BASE58_LEN: usize = 44;

pub fn from_str(s: &str) -> Result<Pubkey, Err> {
    if s.len() > MAX_BASE58_LEN {
        return Err(ParsePubkeyError::WrongSize);
    }
    let pubkey_vec = bs58::decode(s)
        .into_vec()
        .map_err(|_| ParsePubkeyError::Invalid)?;
    if pubkey_vec.len() != mem::size_of::<Pubkey>() {
        Err(ParsePubkeyError::WrongSize)
    } else {
        Pubkey::try_from(pubkey_vec).map_err(|_| ParsePubkeyError::Invalid)
    }
}
pub fn from_Pubkey(pubkey: Pubkey) -> String {
    let pubkey_vec = bs58::encode(pubkey)
        .into_string();
    return pubkey_vec;
}

pub async fn get_tokens_infos(tokens: Vec<TokenInArb>) -> HashMap<String, TokenInfos> {
    let env = Env::new();
    let rpc_client = RpcClient::new(env.rpc_url);

    let mut pubkeys_str: Vec<String> = Vec::new();
    let mut pubkeys: Vec<Pubkey> = Vec::new();
    for token in tokens.clone() {
        pubkeys_str.push(token.address.clone());
        pubkeys.push(from_str(token.address.clone().as_str()).unwrap());
    }
    let batch_results = rpc_client.get_multiple_accounts(&pubkeys).unwrap();

    let mut tokens_infos: HashMap<String, TokenInfos> = HashMap::new();

    for (j, account) in batch_results.iter().enumerate() {
        let account = account.clone().unwrap();
        let mint_layout = MintLayout::try_from_slice(&account.data).unwrap();

        let symbol = tokens.iter().find(|r| *r.address == pubkeys_str[j]).expect("Symbol token not found");
        tokens_infos.insert(pubkeys_str[j].clone(), TokenInfos{
            address: pubkeys_str[j].clone(),
            decimals: mint_layout.decimals,
            symbol: symbol.clone().symbol
        });
    }
    return tokens_infos;

}

pub async fn make_request(req_url: String) -> Result<reqwest::Response, Error> {
    reqwest::get(req_url).await
}

#[derive(BorshDeserialize, Debug)]
pub struct MintLayout {
    pub mint_authority_option: u32,
    pub mint_authority: Pubkey,
    pub supply: u64,
    pub decimals: u8,
    pub is_initialized: bool,
    pub freeze_authority_option: u32,
    pub freeze_authority: Pubkey,
}
