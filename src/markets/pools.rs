use crate::common::constants::Env;
use super::meteora::{fetch_data_meteora, MeteoraDEX};
use super::types::{Dex, DexLabel};
use super::raydium_clmm::{RaydiumClmmDEX, fetch_data_raydium_clmm};
use super::orca::{OrcaDex, fetch_data_orca};
use super::orca_whirpools::{OrcaDexWhirpools, fetch_data_orca_whirpools};
use super::raydium::{fetch_data_raydium, RaydiumDEX};

use strum::IntoEnumIterator;
use std::fs::File;
use std::io::Write;
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use reqwest::get;
use log::info;
use solana_client::rpc_client::RpcClient;


pub async fn load_all_pools(refecth_api: bool) -> Vec<Dex> {
    if refecth_api {
        fetch_data_raydium_clmm().await;
        fetch_data_orca().await;
        fetch_data_orca_whirpools().await;
        fetch_data_raydium().await;
        fetch_data_meteora().await;
    }

    let mut dex1 = Dex::new(DexLabel::RAYDIUM_CLMM);
    let dex_raydium_clmm = RaydiumClmmDEX::new(dex1);
    let mut dex2 = Dex::new(DexLabel::ORCA);
    let dex_orca = OrcaDex::new(dex2);
    let mut dex3 = Dex::new(DexLabel::ORCA_WHIRLPOOLS);
    let dex_orca_whirpools = OrcaDexWhirpools::new(dex3);
    let mut dex4 = Dex::new(DexLabel::RAYDIUM);
    let dex_raydium = RaydiumDEX::new(dex4);
    let mut dex5 = Dex::new(DexLabel::METEORA);
    let dex_meteora = MeteoraDEX::new(dex5);
    
    // println!("random_pair {:?}", dex_raydium.dex.pairToMarkets.get("So11111111111111111111111111111111111111112/mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"));
    
    let mut results: Vec<Dex> = Vec::new();
    results.push(dex_raydium_clmm.dex);
    results.push(dex_orca.dex);
    results.push(dex_orca_whirpools.dex);
    results.push(dex_raydium.dex);
    results.push(dex_meteora.dex);
    return results

}

