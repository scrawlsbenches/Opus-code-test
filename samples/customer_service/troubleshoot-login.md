# Login and Account Access Troubleshooting Guide

## Cannot Sign In - Common Issues

### Forgot Password

**Problem**: You can't remember your password.

**Solution**:
1. Click "Forgot Password" on the login page
2. Enter your registered email address
3. Check your inbox for password reset link (valid for 24 hours)
4. Check spam/junk folder if email doesn't arrive within 5 minutes
5. Click the link and create a new strong password
6. Confirm the new password and sign in

**Note**: For security, password reset links expire after 24 hours and can only be used once.

### Email Not Recognized

**Problem**: System says your email address isn't registered.

**Troubleshooting**:
- Verify you're using the correct email address
- Check if you might have registered with a different email
- Try alternative email addresses you commonly use
- Look for welcome or confirmation emails from us
- Contact support if you're certain you have an account

Common causes:
- Typos in email address
- Multiple accounts with different emails
- Account registered by another team member (enterprise)
- Account deleted due to inactivity (after 3 years)

### Incorrect Password

**Problem**: "Password incorrect" error when signing in.

**Troubleshooting**:
1. Double-check caps lock is off (passwords are case-sensitive)
2. Ensure you're typing the complete password
3. Try copying and pasting from a password manager
4. Look for special characters that might be mistyped
5. Reset your password if still unable to login after 3 attempts

**Security Note**: After 5 failed login attempts, your account is temporarily locked for 30 minutes to prevent unauthorized access.

## Account Locked or Suspended

### Temporary Account Lock

**Problem**: "Account temporarily locked" message.

**Explanation**: This security feature activates after multiple failed login attempts.

**Resolution**:
- Wait 30 minutes and try again
- Use the "Forgot Password" link to reset immediately
- Contact support if you suspect unauthorized access
- Enable two-factor authentication after regaining access

### Account Suspended

**Problem**: "Account suspended" or "Account disabled" error.

**Common Reasons**:
- Payment failure or outstanding balance
- Violation of terms of service
- Security concern or suspicious activity
- Inactivity (no login for 18+ months)
- Request from account owner or administrator

**Resolution**:
1. Check email for suspension notice with details
2. Contact support team with your account details
3. Resolve any outstanding issues (payment, verification, etc.)
4. Request account reactivation if eligible

**Processing Time**: Account reactivation typically takes 1-2 business days after issue resolution.

## Two-Factor Authentication Issues

### Lost or Changed Phone

**Problem**: Can't receive 2FA codes on your device.

**Solutions**:
1. Use backup codes provided during 2FA setup
2. Click "Try another way" on 2FA prompt
3. Receive code via backup email if configured
4. Contact support to temporarily disable 2FA (requires identity verification)

**Prevention**: Always save backup codes when enabling 2FA and keep them in a secure location.

### 2FA Code Not Working

**Problem**: Authentication code is rejected.

**Troubleshooting**:
- Ensure device time is synchronized correctly (critical for TOTP apps)
- Use the most recent code (codes expire every 30 seconds)
- Check you're entering the code for the correct account
- Verify authenticator app is set up for correct service
- Request a new code via SMS or email if available

**Time Sync Issue**: If using an authenticator app, go to app settings and manually sync time. Out-of-sync clocks cause code mismatches.

### Can't Set Up 2FA

**Problem**: Error when trying to enable two-factor authentication.

**Checklist**:
- Phone number is in correct format (include country code)
- Phone number is not already registered to another account
- You have cell service or wifi for receiving codes
- Authenticator app is updated to latest version
- QR code scanner is functioning properly

**Alternative**: If QR scanning fails, use manual entry with the provided secret key.

## Browser and Technical Issues

### Page Won't Load

**Problem**: Login page shows errors or won't load.

**Quick Fixes**:
1. Clear browser cache and cookies
2. Try a different browser (Chrome, Firefox, Safari, Edge)
3. Disable browser extensions temporarily
4. Check internet connection
5. Try incognito/private browsing mode
6. Restart your browser

**Still Not Working**: Check our status page at status.example.com for any ongoing service disruptions.

### Cookies or JavaScript Disabled

**Problem**: "Please enable cookies" or "JavaScript required" message.

**Resolution**:

For Chrome:
1. Settings > Privacy and Security > Cookies
2. Enable "Allow all cookies" or add our site to exceptions
3. Settings > Privacy and Security > Site Settings > JavaScript
4. Ensure JavaScript is "Allowed"

For Firefox:
1. Preferences > Privacy & Security
2. Set "Custom" under Enhanced Tracking Protection
3. Uncheck "Cookies" or add exception for our site
4. Ensure JavaScript is enabled in about:config

### Redirect Loop or Infinite Loading

**Problem**: Login page keeps redirecting or loading indefinitely.

**Fixes**:
1. Clear all browser cookies for our domain
2. Close all browser tabs/windows completely
3. Restart browser
4. Update browser to latest version
5. Try different network (switch from WiFi to cellular or vice versa)

**Advanced**: Check if corporate firewall or VPN is interfering with authentication.

## Enterprise and SSO Login

### Single Sign-On Not Working

**Problem**: SSO redirect fails or doesn't recognize your credentials.

**Troubleshooting**:
- Verify you're using the correct SSO login URL (ask your IT admin)
- Ensure you're logged into your organization's identity provider
- Check if SSO session expired (try logging into other SSO apps)
- Confirm your email domain matches SSO configuration
- Contact your IT administrator for SSO troubleshooting

**Common SSO Issues**:
- User not provisioned in identity provider
- Email address mismatch between systems
- Expired certificates or metadata
- Firewall blocking authentication endpoints

### Cannot Access Team Account

**Problem**: Can't log into your organization's account.

**Checklist**:
- Verify invitation email was accepted
- Check with account administrator about access permissions
- Ensure you're using work email, not personal email
- Confirm account administrator added you to the team
- Check if your role has login permissions (some roles are API-only)

**Getting Help**: Contact your organization's account administrator first, as they control team access.

## Mobile App Login Issues

### App Won't Accept Credentials

**Problem**: Login works on web but not in mobile app.

**Solutions**:
1. Verify you're using the official app (check app store)
2. Update app to latest version
3. Clear app cache (Settings > Storage > Clear Cache)
4. Uninstall and reinstall the app
5. Try web browser on mobile device to isolate issue
6. Check if app needs specific permissions (contacts, notifications)

### Biometric Login Failed

**Problem**: Fingerprint or Face ID authentication not working.

**Troubleshooting**:
- Ensure biometric authentication is enabled in phone settings
- Re-register biometric credentials in app settings
- Verify your device supports the biometric method
- Check device biometrics work in other apps
- Fall back to password login and reconfigure biometrics

## Getting Additional Help

If you've tried these troubleshooting steps and still can't access your account:

**Contact Support**:
- Email: support@example.com (response within 24 hours)
- Live Chat: Available Mon-Fri 9am-5pm EST
- Phone: 1-800-SUPPORT (1-800-787-7678)

**Include in Your Request**:
- Email address or username
- Description of the problem
- Error messages (screenshot if possible)
- Steps you've already tried
- Browser and operating system version

**Priority Support**: Enterprise customers can escalate login issues through their dedicated account manager for faster resolution (SLA: 4-hour response time).
