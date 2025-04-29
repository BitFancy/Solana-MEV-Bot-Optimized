use crate::errors::LBError;
use crate::math::safe_math::SafeMath;
use crate::state::lb_pair::LbPair;
use anchor_lang::prelude::*;

#[derive(Accounts)]
pub struct SetPreActivationInfo<'info> {
    #[account(
        mut,
        has_one = creator
    )]
    pub lb_pair: AccountLoader<'info, LbPair>,

    pub creator: Signer<'info>,
}

pub fn handle(
    ctx: Context<SetPreActivationInfo>,
    pre_activation_slot_duration: u16, // Around 9 hours buffer
) -> Result<()> {
    Ok(())
}
