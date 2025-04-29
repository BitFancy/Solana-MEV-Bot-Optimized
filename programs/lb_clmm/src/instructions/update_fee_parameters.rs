use crate::assert_eq_admin;
use crate::errors::LBError;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct FeeParameter {
    /// Portion of swap fees retained by the protocol by controlling protocol_share parameter. protocol_swap_fee = protocol_share * total_swap_fee
    pub protocol_share: u16,
    /// Base factor for base fee rate
    pub base_factor: u16,
}

#[event_cpi]
#[derive(Accounts)]
pub struct UpdateFeeParameters<'info> {
    #[account(mut)]
    pub lb_pair: AccountLoader<'info, LbPair>,

    #[account(constraint = assert_eq_admin(admin.key()) @ LBError::InvalidAdmin)]
    pub admin: Signer<'info>,
}

pub fn handle(ctx: Context<UpdateFeeParameters>, fee_parameter: FeeParameter) -> Result<()> {
    Ok(())
}
