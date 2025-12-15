# Payment Processing Troubleshooting Guide

## Payment Declined or Failed

### Credit Card Declined

**Problem**: Your credit card payment is rejected at checkout.

**Common Reasons and Solutions**:

1. **Insufficient Funds**
   - Check account balance includes available credit
   - Pay down balance and retry
   - Use alternative payment method

2. **Incorrect Card Information**
   - Verify card number is entered correctly (no spaces)
   - Check expiration date (MM/YY format)
   - Confirm CVV/CVC security code (3-4 digits on back)
   - Ensure billing address matches card on file with bank

3. **Expired Card**
   - Check expiration date on physical card
   - Contact card issuer for replacement
   - Update payment method with new card details

4. **Card Security Block**
   - Bank may flag transaction as potentially fraudulent
   - Contact your bank to authorize the charge
   - Inform bank it's a legitimate purchase
   - Retry payment after bank authorization

5. **Daily Transaction Limit Reached**
   - Many cards have daily spending limits
   - Wait 24 hours or contact bank to increase limit
   - Split payment across multiple cards if supported

### Bank Transfer or ACH Failed

**Problem**: Direct bank payment doesn't process.

**Troubleshooting**:
- Verify account and routing numbers are correct
- Ensure sufficient funds in account
- Check if account allows ACH debits
- Confirm account is active and not frozen
- Allow 3-5 business days for ACH processing

**Common ACH Errors**:
- R01: Insufficient funds
- R02: Account closed
- R03: Invalid account number
- R04: Invalid account type
- R10: Account holder deceased

Contact your bank if error code persists after verification.

### PayPal or Digital Wallet Issues

**Problem**: PayPal, Apple Pay, or Google Pay transaction fails.

**Solutions**:

**PayPal**:
1. Verify PayPal account is active and verified
2. Check PayPal balance or linked funding source
3. Remove and re-add payment method in PayPal
4. Clear PayPal authorization and reconnect
5. Contact PayPal support for account-specific issues

**Apple Pay**:
1. Ensure card is active in Apple Wallet
2. Verify device supports Apple Pay (iPhone 6+, Watch Series 1+)
3. Check Face ID/Touch ID is enabled
4. Update iOS to latest version
5. Remove and re-add card to wallet

**Google Pay**:
1. Confirm card is added to Google Pay
2. Check NFC is enabled on Android device
3. Ensure screen lock is configured
4. Update Google Pay app
5. Verify merchant accepts Google Pay

## Payment Processing Errors

### "Payment Gateway Error" or "Processing Failed"

**Problem**: Generic payment error message appears.

**Immediate Actions**:
1. Wait 5 minutes and retry (temporary server issues)
2. Use a different payment method
3. Try a different browser or device
4. Disable browser extensions (ad blockers, privacy tools)
5. Clear browser cookies and cache

**If Error Persists**:
- Check our status page for system outages
- Try alternative checkout flow (guest checkout vs. signed-in)
- Contact support with exact error message and timestamp

### "Payment Already Processed" Warning

**Problem**: Multiple payment attempts may have succeeded.

**Important**: Do not retry payment multiple times immediately.

**What to Do**:
1. Check email for payment confirmation
2. Review your bank/card statement
3. Log into account to verify order status
4. If duplicate charges appear, contact support immediately
5. Allow 3-5 business days for duplicate charges to reverse automatically

**Prevention**: Wait at least 2 minutes between payment retry attempts.

### Currency Conversion Issues

**Problem**: Unexpected currency or conversion rate applied.

**Understanding Currency Processing**:
- Charges appear in your local currency or merchant currency
- Conversion rates fluctuate based on market conditions
- Some banks charge foreign transaction fees (1-3%)
- Display currency may differ from settlement currency

**Solutions**:
- Check if merchant supports your native currency
- Consider using multi-currency credit card
- Contact card issuer about conversion rates and fees
- For large purchases, bank transfers may offer better rates

## Promotional Code and Discount Problems

### Coupon Code Not Working

**Problem**: Discount code is rejected or doesn't apply.

**Common Reasons**:

1. **Expired Code**
   - Check promotion end date
   - Request updated code if available
   - Sign up for notifications about new promotions

2. **Minimum Purchase Not Met**
   - Many codes require minimum cart value
   - Add items to reach threshold
   - Check code terms and conditions

3. **Restricted Items**
   - Some codes exclude sale items, specific brands, or categories
   - Verify eligible products
   - Remove excluded items and reapply code

4. **First-Time Customer Only**
   - Code may be limited to new accounts
   - Check if you've used the code before
   - Try different promotional offers

5. **One Use Per Customer**
   - Most codes can only be used once per account
   - System blocks previously used codes
   - Contact support if you believe error occurred

**How to Apply Codes**:
1. Add items to cart
2. Proceed to checkout
3. Find "Promo Code" or "Discount Code" field
4. Enter code exactly as provided (case-sensitive)
5. Click "Apply" or "Redeem"
6. Verify discount reflects in order total

### Gift Card Balance Issues

**Problem**: Gift card doesn't cover purchase or shows wrong balance.

**Troubleshooting**:
- Check gift card balance before checkout (Account > Gift Cards)
- Verify gift card is activated (physical cards need activation)
- Ensure gift card hasn't expired (check terms)
- Confirm you're entering the complete card number and PIN
- Check if gift card is region-specific

**Partial Payment with Gift Card**:
If gift card doesn't cover full amount:
1. Apply gift card first
2. Remaining balance charged to credit card
3. System prompts for second payment method automatically

## Recurring Payment and Subscription Issues

### Auto-Renewal Failed

**Problem**: Subscription payment didn't process automatically.

**Why This Happens**:
- Card on file expired
- Insufficient funds at renewal time
- Card was canceled or reported lost
- Bank declined recurring charge
- Payment method removed from account

**Resolution**:
1. Update payment information in account settings
2. Manually process payment for current period
3. Verify auto-renewal is enabled
4. Check email for renewal reminder notices

**Grace Period**: Most subscriptions have 7-day grace period before service interruption.

### Cannot Cancel Recurring Payment

**Problem**: Unable to stop auto-renewal or recurring charge.

**Steps to Cancel**:
1. Log into account
2. Navigate to Subscriptions or Billing
3. Select active subscription
4. Click "Cancel Subscription" or "Turn Off Auto-Renewal"
5. Confirm cancellation
6. Save confirmation email

**If Option Not Available**:
- Verify you have account administrator permissions
- Check if subscription is managed by third party (iTunes, Google Play)
- Contact support to cancel manually
- As last resort, remove payment method from account

**Note**: Canceling auto-renewal maintains access through current billing period.

### Charged After Cancellation

**Problem**: Payment processed despite canceling subscription.

**Possible Explanations**:
- Cancellation wasn't confirmed (check confirmation email)
- Canceled after billing cycle cutoff (timing issue)
- Separate subscription or service charged
- Trial period ended and converted to paid

**What to Do**:
1. Verify cancellation date vs. billing date
2. Check all active subscriptions
3. Review cancellation confirmation
4. Request refund if charge was erroneous
5. Confirm all subscriptions are now canceled

## Security and Fraud Concerns

### Suspicious Charge Investigation

**Problem**: Unrecognized charge from our company.

**Verification Steps**:
1. Check if family member or colleague made purchase
2. Review all email accounts for order confirmations
3. Check if charge description varies from company name
4. Verify date matches any orders or subscriptions
5. Contact support with transaction details

**Reporting Fraud**:
If charge is confirmed fraudulent:
- Contact our fraud department immediately
- File dispute with your bank/card issuer
- Change account password
- Review account activity for unauthorized access
- Enable two-factor authentication

### Payment Information Security

**Problem**: Concerns about payment data security.

**Our Security Measures**:
- PCI DSS Level 1 compliant (highest security standard)
- End-to-end encryption for all transactions
- Tokenization - we don't store full card numbers
- Regular security audits and penetration testing
- Fraud detection and prevention systems

**Your Security Best Practices**:
- Never share account password
- Use strong, unique passwords
- Enable two-factor authentication
- Monitor account for suspicious activity
- Only make payments on secure networks (avoid public WiFi)
- Verify URL shows https:// and padlock icon

## Getting Payment Support

**Before Contacting Support, Have Ready**:
- Order number or transaction ID
- Payment method used (last 4 digits of card)
- Exact error message or screenshot
- Date and time of attempted payment
- Browser and device information

**Contact Methods**:
- Live Chat: Instant support during business hours
- Email: billing@example.com (24-48 hour response)
- Phone: 1-800-BILLING (1-800-245-5464)
- Support Ticket: Submit through account portal

**Enterprise Customers**:
Contact your dedicated account manager or use priority support channel. SLA guarantees 4-hour response for payment issues affecting service access.

**Escalation**:
If standard support doesn't resolve your payment issue within 48 hours, request escalation to billing supervisor or dispute resolution team.
