use log::{error, info};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{commitment_config::CommitmentConfig, pubkey::Pubkey, signature::Signature};
use anyhow::{format_err, Result};
use std::{
    borrow::Cow,
    convert::{identity, TryFrom},
    mem::size_of,
    thread::{self, sleep}, time::{self, Duration, Instant},
};
use safe_transmute::{
    to_bytes::{transmute_one_to_bytes, transmute_to_bytes},
    transmute_many_pedantic, transmute_one_pedantic,
};
use serum_dex::state::{gen_vault_signer_key, AccountFlag, Market, MarketState, MarketStateV2};

use crate::common::constants::Env;

use super::create_transaction::ChainType;


pub async fn check_tx_status(commitment_config: CommitmentConfig, chain: ChainType ,signature: Signature) -> Result<bool> {
    let env = Env::new();
    let rpc_url = if chain.clone() == ChainType::Mainnet { env.rpc_url } else { env.devnet_rpc_url };
    let rpc_client: RpcClient = RpcClient::new(rpc_url);

    let start = Instant::now();
    let mut counter = 0;
    loop {
        let confirmed = rpc_client.confirm_transaction(&signature)?;
        info!("Is confirmed? {:?}", confirmed);

        let status = rpc_client.get_signature_status_with_commitment_and_history(&signature, commitment_config, true)?;
        println!("Status: {:?}", status);

        let seventy_secs = Duration::from_secs(11);
        if confirmed {
            info!("✅ Transaction Confirmed with Confirmation");
            info!("Status {:?}", status.clone().unwrap());
            return Ok(true);
        }
        if status.is_some() {
            info!("✅ Transaction Confirmed with Status");
            return Ok(true);
        }
        if start.elapsed() >= seventy_secs {
            error!("❌ Transaction not confirmed");
            return Ok(false);
        }
        let ten_secs = Duration::from_secs(10);
        info!("⏳ {} seconds...", 10 * counter);
        sleep(ten_secs);
        counter += 1;
    }
}

#[cfg(target_endian = "little")]
fn remove_dex_account_padding<'a>(data: &'a [u8]) -> Result<Cow<'a, [u64]>> {
    use serum_dex::state::{ACCOUNT_HEAD_PADDING, ACCOUNT_TAIL_PADDING};
    let head = &data[..ACCOUNT_HEAD_PADDING.len()];
    if data.len() < ACCOUNT_HEAD_PADDING.len() + ACCOUNT_TAIL_PADDING.len() {
        return Err(format_err!(
            "dex account length {} is too small to contain valid padding",
            data.len()
        ));
    }
    if head != ACCOUNT_HEAD_PADDING {
        return Err(format_err!("dex account head padding mismatch"));
    }
    let tail = &data[data.len() - ACCOUNT_TAIL_PADDING.len()..];
    if tail != ACCOUNT_TAIL_PADDING {
        return Err(format_err!("dex account tail padding mismatch"));
    }
    let inner_data_range = ACCOUNT_HEAD_PADDING.len()..(data.len() - ACCOUNT_TAIL_PADDING.len());
    let inner: &'a [u8] = &data[inner_data_range];
    let words: Cow<'a, [u64]> = match transmute_many_pedantic::<u64>(inner) {
        Ok(word_slice) => Cow::Borrowed(word_slice),
        Err(transmute_error) => {
            let word_vec = transmute_error.copy().map_err(|e| e.without_src())?;
            Cow::Owned(word_vec)
        }
    };
    Ok(words)
}

#[cfg(target_endian = "little")]
pub fn get_keys_for_market<'a>(
    client: &'a RpcClient,
    program_id: &'a Pubkey,
    market: &'a Pubkey,
) -> Result<MarketPubkeys> {

    let account_data: Vec<u8> = client.get_account_data(&market)?;
    let words: Cow<[u64]> = remove_dex_account_padding(&account_data)?;
    let market_state: MarketState = {
        let account_flags = Market::account_flags(&account_data)?;
        if account_flags.intersects(AccountFlag::Permissioned) {
            // println!("MarketStateV2");
            let state = transmute_one_pedantic::<MarketStateV2>(transmute_to_bytes(&words))
                .map_err(|e| e.without_src())?;
            state.check_flags(true)?;
            state.inner
        } else {
            // println!("MarketStateV");
            let state = transmute_one_pedantic::<MarketState>(transmute_to_bytes(&words))
                .map_err(|e| e.without_src())?;
            state.check_flags(true)?;
            state
        }
    };
    let vault_signer_key =
        gen_vault_signer_key(market_state.vault_signer_nonce, market, program_id)?;
    assert_eq!(
        transmute_to_bytes(&identity(market_state.own_address)),
        market.as_ref()
    );
    Ok(MarketPubkeys {
        market: Box::new(*market),
        req_q: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.req_q))).unwrap(),
        ),
        event_q: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.event_q))).unwrap(),
        ),
        bids: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.bids))).unwrap(),
        ),
        asks: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.asks))).unwrap(),
        ),
        coin_vault: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.coin_vault))).unwrap(),
        ),
        pc_vault: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.pc_vault))).unwrap(),
        ),
        vault_signer_key: Box::new(vault_signer_key),
        coin_mint: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.coin_mint))).unwrap(),
        ),
        pc_mint: Box::new(
            Pubkey::try_from(transmute_one_to_bytes(&identity(market_state.pc_mint))).unwrap(),
        ),
        coin_lot_size: market_state.coin_lot_size,
        pc_lot_size: market_state.pc_lot_size,
    })
}

#[derive(Debug)]
pub struct MarketPubkeys {
    pub market: Box<Pubkey>,
    pub req_q: Box<Pubkey>,
    pub event_q: Box<Pubkey>,
    pub bids: Box<Pubkey>,
    pub asks: Box<Pubkey>,
    pub coin_vault: Box<Pubkey>,
    pub pc_vault: Box<Pubkey>,
    pub vault_signer_key: Box<Pubkey>,
    pub coin_mint: Box<Pubkey>,
    pub pc_mint: Box<Pubkey>,
    pub coin_lot_size: u64,
    pub pc_lot_size: u64,
}

pub fn average(numbers: Vec<u64>) -> u64 {
    let sum: u64 = numbers.iter().sum();
    let count = numbers.len() as u64;
    sum / count
}