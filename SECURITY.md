# Security Policy

## Supported Versions

HAMLET is currently in active development. Security updates are provided for the latest version only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in HAMLET, please report it responsibly.

### How to Report

**Please DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, report security issues by email to:
- **Email**: john@foundryside.dev
- **Subject**: [SECURITY] Brief description of vulnerability

### What to Include

Please include the following information in your report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: What could an attacker accomplish?
3. **Affected Components**: Which parts of HAMLET are affected?
4. **Reproduction Steps**: Step-by-step instructions to reproduce
5. **Proof of Concept**: Code or commands demonstrating the issue (if applicable)
6. **Suggested Fix**: If you have ideas on how to fix it
7. **Your Details**: Name and contact info for acknowledgment (optional)

### Example Report

```
Subject: [SECURITY] Arbitrary Code Execution via Malicious Config Files

Description:
HAMLET loads YAML config files without proper validation, allowing an attacker
to execute arbitrary Python code via PyYAML's unsafe load() function.

Impact:
An attacker who can provide a malicious config file could execute arbitrary
code on the system running HAMLET.

Affected Components:
- src/townlet/environment/config_loader.py (line 45)

Reproduction Steps:
1. Create a malicious config file with PyYAML gadgets
2. Run: python scripts/run_demo.py --config malicious.yaml
3. Observe code execution

Suggested Fix:
Use yaml.safe_load() instead of yaml.load()
```

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Vulnerability Assessment**: Within 5 business days
- **Fix Development**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next minor release
- **Public Disclosure**: After fix is released and users have time to update

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Investigation**: We'll verify and assess the vulnerability
3. **Communication**: We'll keep you updated on our progress
4. **Fix Development**: We'll develop and test a fix
5. **Release**: We'll release a patched version
6. **Disclosure**: We'll publicly disclose the vulnerability (with your permission)
7. **Credit**: We'll credit you in the CHANGELOG and security advisory (if you wish)

## Security Best Practices

When using HAMLET, follow these best practices:

### Configuration Files

- **Validate configs**: Only load configs from trusted sources
- **Avoid user-supplied configs**: Don't load configs directly from user input without validation
- **Review YAML files**: Inspect config files before using them

### Training Data

- **Sanitize inputs**: Validate all external data before use
- **Checkpoint integrity**: Verify checkpoint integrity before loading
- **Isolated environments**: Train in isolated environments when possible

### Deployment

- **Latest version**: Use the latest version of HAMLET
- **Updated dependencies**: Keep PyTorch, NumPy, and other dependencies up-to-date
- **Minimal permissions**: Run HAMLET with minimal required permissions
- **Network isolation**: Isolate training environments from production networks

### Development

- **Code review**: Review all code changes for security implications
- **Dependency scanning**: Regularly scan dependencies for vulnerabilities
- **Static analysis**: Use tools like Bandit for Python security checks
- **Input validation**: Validate all user inputs and external data

## Known Security Considerations

### PyTorch Model Loading

HAMLET uses PyTorch's `torch.load()` for loading model checkpoints. **Only load checkpoints from trusted sources**, as PyTorch uses pickle which can execute arbitrary code.

### YAML Config Loading

HAMLET loads configuration files using PyYAML. We use `yaml.safe_load()` to prevent code execution vulnerabilities. Always verify config files come from trusted sources.

### WebSocket Inference Server

The live inference server (port 8766) is intended for local use only. Do not expose it directly to the internet without proper authentication and encryption.

### User-Supplied Affordances

If extending HAMLET to allow user-defined affordances or game mechanics, validate all custom logic thoroughly to prevent code injection.

## Security Updates

Security updates will be announced via:
1. GitHub Security Advisories
2. CHANGELOG.md with [SECURITY] prefix
3. Release notes

## Out of Scope

The following are explicitly out of scope for security reports:

- Denial of Service via resource exhaustion (training RL models is resource-intensive by design)
- Issues requiring physical access to the machine
- Social engineering attacks
- Issues in third-party dependencies (report to those projects directly)
- Pedagogical "interesting failures" (e.g., reward hacking) - these are features, not bugs!

## Questions?

For non-security questions about HAMLET, please:
- Open a GitHub Discussion
- Create a public GitHub Issue
- Read the documentation in `docs/`

For security-related questions (not vulnerabilities), you can email john@foundryside.dev with subject line "[SECURITY QUESTION]".

---

Thank you for helping keep HAMLET and its users safe!
