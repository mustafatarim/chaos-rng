# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

I take security seriously. If you discover a security vulnerability, please follow these steps:

### Private Disclosure

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send an email to: **mail@mustafatarim.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)

### What to Expect

- **Initial Response**: We will acknowledge your email within 48 hours
- **Investigation**: We will investigate and validate the vulnerability within 5 business days
- **Resolution**: We will work to resolve confirmed vulnerabilities as quickly as possible
- **Disclosure**: We will coordinate public disclosure after a fix is available

### Security Considerations for Chaos RNG

When reporting vulnerabilities, please consider these specific aspects of our library:

#### Cryptographic Security
- Seed generation and handling
- Random number quality and predictability
- Side-channel attacks
- State recovery vulnerabilities

#### Implementation Security
- Memory safety issues
- Buffer overflows or underflows
- Integer overflow/underflow
- Denial of service vulnerabilities

#### Dependency Security
- Vulnerabilities in third-party dependencies
- Supply chain security issues

#### Mathematical Security
- Weaknesses in the chaotic system implementation
- Statistical biases in output
- Correlation attacks
- Period detection vulnerabilities

## Security Best Practices for Users

### Secure Usage
1. **Proper Seeding**: Always use cryptographically secure seeds for security-critical applications
2. **Regular Updates**: Keep the library updated to the latest version
3. **Environment Security**: Protect the environment where random numbers are generated
4. **State Protection**: Don't expose internal generator state

### Example Secure Usage
```python
import secrets
from chaos_rng import ThreeBodyRNG

# Use secure seeding for cryptographic applications
secure_seed = secrets.randbits(256)
rng = ThreeBodyRNG(seed=secure_seed, secure=True)

# Enable periodic reseeding for long-running applications
rng.enable_reseeding(interval=1000000)  # Reseed every 1M samples
```

### What NOT to Do
- Don't use predictable seeds (like timestamps) for security applications
- Don't reuse the same generator state across security boundaries
- Don't rely solely on chaos-based entropy for life-critical systems
- Don't disable security features unless you understand the implications

## Security Features

### Built-in Security Measures
- **Secure Seeding**: Optional cryptographically secure seed generation
- **Forward Secrecy**: Automatic reseeding prevents state recovery
- **Statistical Validation**: Built-in statistical tests ensure randomness quality
- **Thread Safety**: Safe for multi-threaded applications

### Security Testing
We regularly perform:
- **Static Analysis**: Automated security scanning with Bandit
- **Dependency Auditing**: Regular checks for vulnerable dependencies
- **Fuzzing**: Input validation testing
- **Statistical Testing**: NIST SP 800-22 randomness tests

## Responsible Disclosure

We appreciate the security research community's efforts to improve our library's security. 

### Recognition
- We will acknowledge security researchers who responsibly disclose vulnerabilities
- With your permission, we will credit you in our security advisories
- We may include you in our contributors list

### Coordinated Disclosure Timeline
1. **Day 0**: Vulnerability reported
2. **Day 1-5**: Initial assessment and validation
3. **Day 6-30**: Development of fix
4. **Day 31-45**: Testing and validation of fix
5. **Day 46-60**: Coordinated public disclosure

### Security Updates
Security fixes will be:
- Released as soon as possible
- Clearly marked in release notes
- Accompanied by a security advisory
- Backported to supported versions when feasible

## Contact Information

- **Security Email**: mail@mustafatarim.com
- **GPG Key**: Available upon request
- **General Contact**: mail@mustafatarim.com

## Additional Resources

- [NIST SP 800-22 Statistical Tests](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)
- [Cryptographically Secure Pseudorandom Number Generators](https://en.wikipedia.org/wiki/Cryptographically_secure_pseudorandom_number_generator)
- [Python Security Best Practices](https://python.org/dev/security/)
