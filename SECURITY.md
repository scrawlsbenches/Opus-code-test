# Security Policy

## Reporting Security Vulnerabilities

We take the security of the Cortical Text Processor seriously. If you discover a security vulnerability, please report it privately to help us address it responsibly.

**To report a security issue:**

1. **Do not** open a public GitHub issue
2. Send details to the project maintainers via private communication
3. Include:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if available)

We will acknowledge receipt of your report and work with you to understand and address the issue.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

We support the latest version of the Cortical Text Processor. Security updates and patches are applied to the current release.

## Platform Support

The Cortical Text Processor is designed for **Linux and macOS only**.

**Supported Platforms:**
- Linux (all distributions)
- macOS (all versions)

**Unsupported Platforms:**
- Windows - Not supported due to POSIX-specific file locking (`fcntl.flock()`)

If you require Windows support, please open a feature request, though this would require significant architectural changes.

## Security Design Principles

The Cortical Text Processor is built with security as a core design principle:

### 1. Zero External Dependencies

- **No third-party runtime dependencies** - eliminates supply chain attack vectors
- Pure Python implementation using only standard library
- Development dependencies (pytest, coverage) are isolated to testing

**Security benefit:** No exposure to vulnerabilities in external packages.

### 2. JSON-First Persistence (Pickle Deprecated)

- **Default storage format: JSON** - human-readable, no code execution risk
- **Pickle format deprecated** - marked for removal due to Remote Code Execution (RCE) vulnerability
- All new code uses JSON serialization

**Security benefit:** Loading saved data cannot execute arbitrary code.

**Migration from pickle:**
```python
from cortical.processor import CorticalTextProcessor

# Load from legacy pickle (with deprecation warning)
processor = CorticalTextProcessor.load('corpus.pkl')

# Save as secure JSON
processor.save('corpus.json')
```

### 3. Atomic File Writes

- All file writes use atomic operations via temporary files + rename
- Prevents partial writes and corruption from crashes
- Transaction-safe state updates

**Security benefit:** Corruption cannot leave the system in an exploitable state.

### 4. Process-Safe File Locking

- POSIX `fcntl.flock()` for exclusive file access
- Prevents concurrent modification races
- Lock acquisition with timeout detection

**Security benefit:** Multi-process safety without data corruption.

### 5. No Network Access Required

- Fully offline operation
- No external API calls
- No telemetry or data collection (except opt-in ML training data stored locally)

**Security benefit:** No exposure to network-based attacks or data exfiltration.

## Known Limitations

### Platform Limitations

- **Windows not supported** - Uses POSIX-specific `fcntl.flock()` which is unavailable on Windows
- Attempting to use on Windows will raise `AttributeError` or `ImportError`

### Deprecated Features

- **Pickle format** - Deprecated and will be removed in a future version
  - Security risk: Can execute arbitrary code during deserialization
  - Mitigation: Migrate to JSON format
  - Timeline: Will be removed in version 2.0

### File System Requirements

- Requires file system supporting atomic renames (POSIX compliance)
- Requires support for exclusive file locks (`fcntl.flock()`)

## Security Best Practices for Users

When using the Cortical Text Processor:

1. **Use JSON format** - Always use `processor.save('path')` without `.pkl` extension
2. **Validate input** - Sanitize user-provided document IDs and text content
3. **Limit corpus size** - Consider memory and disk limits for untrusted content
4. **File permissions** - Use appropriate file permissions for saved corpus data
5. **Audit dependencies** - Regularly check development dependencies for vulnerabilities

## Security Updates

Security updates are released as patch versions and documented in release notes. Monitor the repository for security advisories.

To check your version:
```python
from cortical import __version__
print(__version__)
```

## Responsible Disclosure

We appreciate responsible disclosure and will:
- Acknowledge your report
- Keep you informed of progress
- Credit you in release notes (if desired)
- Work to address issues promptly

Thank you for helping keep the Cortical Text Processor secure.
