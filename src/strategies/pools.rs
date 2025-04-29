use std::{collections::HashMap, thread::sleep, time};
use log::info;
use solana_client::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use crate::{
    arbitrage::types::TokenInArb, common::
        constants::Env
    , markets::{meteora::fetch_new_meteora_pools, orca_whirpools::fetch_new_orca_whirpools, raydium::fetch_new_raydium_pools, types::Market} 
};

pub async fn get_fresh_pools(tokens: Vec<TokenInArb>) -> HashMap<String, Market> {
    let env = Env::new();
    let rpc_client = RpcClient::new_with_commitment(env.rpc_url.as_str(), CommitmentConfig::confirmed());
    let mut new_markets: HashMap<String, Market> = HashMap::new();
    let mut count_new_pools = 0;

    println!("Tokens: {:#?}", tokens);
    for (i, token) in tokens.iter().enumerate() {
        // Avoid fetch for the first token (often SOL)
        if i == 0 {
            continue;
        }
        //Orca Whirpools 
        println!("1 GetProgramAccounts Orca");
        let orca_res_tokena = fetch_new_orca_whirpools(&rpc_client, token.address.clone(), false).await;
        for orca_pool in orca_res_tokena {
            new_markets.insert(orca_pool.0.to_string(), orca_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
        println!("1 GetProgramAccounts Orca");

        let orca_res_tokenb = fetch_new_orca_whirpools(&rpc_client, token.address.clone(), true).await;
        for orca_pool in orca_res_tokenb {
            new_markets.insert(orca_pool.0.to_string(), orca_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
        println!("1 GetProgramAccounts Raydium");
        //Raydium Markets 
        let raydium_res_tokena = fetch_new_raydium_pools(&rpc_client, token.address.clone(), false).await;
        for raydium_pool in raydium_res_tokena {
            new_markets.insert(raydium_pool.0.to_string(), raydium_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
        println!("1 GetProgramAccounts Raydium");
        let raydium_res_tokenb = fetch_new_raydium_pools(&rpc_client, token.address.clone(), true).await;
        for raydium_pool in raydium_res_tokenb {
            new_markets.insert(raydium_pool.0.to_string(), raydium_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
        println!("1 GetProgramAccounts Meteora");
        //Meteora Markets 
        let meteora_res_tokena = fetch_new_meteora_pools(&rpc_client, token.address.clone(), false).await;
        for meteora_pool in meteora_res_tokena {
            new_markets.insert(meteora_pool.0.to_string(), meteora_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
        println!("1 GetProgramAccounts Meteora");
        let meteora_res_tokenb = fetch_new_meteora_pools(&rpc_client, token.address.clone(), true).await;
        for meteora_pool in meteora_res_tokenb {
            new_markets.insert(meteora_pool.0.to_string(), meteora_pool.1);
            count_new_pools += 1;
        }
        sleep(time::Duration::from_millis(2000));
    }
    info!("⚠️⚠️ NO RAYDIUM_CLMM fresh pools !");
    info!("⚠️⚠️ NO ORCA fresh pools !");
    return new_markets;
}