
use std::collections::HashMap;
use anyhow::Result;
use log::info;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use crate::{
    common::{
        constants::Env,
        utils::from_str,
    }, 
    markets::types::{DexLabel, Market}
};

//Get fresh data on all acounts with getMultipleAccounts
pub async fn get_fresh_accounts_states(mut accounts: HashMap<String, Market>) -> HashMap<String, Market> {
    let env = Env::new();
    let rpc_client = RpcClient::new(env.rpc_url);
    let mut counter_fresh_markets = 0;

    let mut markets_vec: Vec<Market> = Vec::new();
    let mut key_vec: Vec<String> = Vec::new();
    let mut pubkeys_vec: Vec<Pubkey> = Vec::new();
    for (key, market) in accounts.clone().iter() {
        markets_vec.push(market.clone());
        key_vec.push(key.clone());
        pubkeys_vec.push(from_str(market.clone().id.as_str()).unwrap());
    }

    for i in (0..pubkeys_vec.len()).step_by(100) {
        let maxLength = std::cmp::min(i + 100, pubkeys_vec.len());
        let batch = &pubkeys_vec[(i..maxLength)];
        
        let batch_results = rpc_client.get_multiple_accounts(&batch).unwrap();
        // println!("BatchResult {:?}", batch_results);
        for (j, account) in batch_results.iter().enumerate() {
            let account = account.clone().unwrap();
            // println!("WhirpoolAccount: {:?}", data);
            let account_data = account.data;

            markets_vec[j].account_data = Some(account_data);
            markets_vec[j].id = key_vec[j].clone();
            counter_fresh_markets += 1;
            accounts.insert(key_vec[j].clone(), markets_vec[j].clone());
        }
    }

    info!("💦💦 Fresh data for {:?} markets", counter_fresh_markets);
    return accounts;
}

