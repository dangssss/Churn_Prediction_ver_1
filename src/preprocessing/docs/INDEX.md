# Documentation Index / Mục Lục Tài Liệu

## 📚 Main Documentation

### 1. **[RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)** — Start Here! 🌟
   - Executive summary of all changes
   - Before/after comparisons
   - Key innovations and results
   - Impact analysis
   - **Best for**: Getting quick overview of what was done

### 2. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** — Architecture Guide
   - Complete project structure with diagrams
   - Layered architecture explanation
   - Layer responsibilities (infrastructure, application, interfaces)
   - File organization and purposes
   - Configuration system
   - Migration guide from old to new structure
   - **Best for**: Understanding the new architecture

### 3. **[WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md)** — Technical Deep-Dive
   - Performance analysis (before/after)
   - Optimization algorithm explained
   - Implementation details
   - Function-level documentation
   - Configuration parameters
   - Execution flow and complexity analysis
   - Edge cases and safety considerations
   - Future enhancements
   - Logging and monitoring
   - **Best for**: Understanding how optimization works

### 4. **[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)** — Rollout Plan
   - Completed tasks (Phase 1)
   - TODO items (Phases 2-5)
   - Detailed task breakdowns
   - Success criteria
   - Risk assessment
   - Migration timeline
   - **Best for**: Planning next steps and rollout

---

## 🎓 Coding Conventions / Quy Ước Lập Trình

Refer to the `conventions/` folder for coding standards:

### Core Conventions
- **[00-Index_and_glossary.md](conventions/00-Index_and_glossary.md)** — Index and terminology
- **[01-Structure_conventions.md](conventions/01-Structure_conventions.md)** — Project structure and layering
- **[02-Config_conventions.md](conventions/02-Config_conventions.md)** — Configuration management
- **[03-Naming_style_conventions.md](conventions/03-Naming_style_conventions.md)** — Naming standards
- **[04-Dependencies_import_conventions.md](conventions/04-Dependencies_import_conventions.md)** — Import organization

### Advanced Topics
- **[05-Error_handling_convention.md](conventions/05-Error_handling_convention.md)** — Error handling patterns
- **[06-Logging_observability_convention.md](conventions/06-Logging_observability_convention.md)** — Logging standards
- **[07-Testing_convention.md](conventions/07-Testing_convention.md)** — Testing requirements
- **[08-Security_secrets_conventions.md](conventions/08-Security_secrets_conventions.md)** — Security practices
- **[09-Git_pr_release_convention.md](conventions/09-Git_pr_release_convention.md)** — Git workflow

### Domain-Specific
- **[10-Code_design_principles.md](conventions/10-Code_design_principles.md)** — Design principles
- **[12-Api_conventions.md](conventions/12-Api_conventions.md)** — API design
- **[13-data_ml_conventions.md](conventions/13-data_ml_conventions.md)** — Data & ML conventions
- **[16-System_architecture_conventions.md](conventions/16-System_architecture_conventions.md)** — Architecture patterns

---

## 🚀 Quick Start Guide

### For New Team Members
1. Start with [RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md) — Get the big picture
2. Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Understand the architecture
3. Check [../config/app_config.py](../config/app_config.py) — See configuration options

### For Developers Working on Features
1. Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#-layered-architecture) — Know where to put code
2. Follow [conventions/03-Naming_style_conventions.md](conventions/03-Naming_style_conventions.md) — Use consistent naming
3. Check [WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md) — Understand optimization

### For DevOps/Deployment
1. Read [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) — Follow deployment phases
2. Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#-centralized-configuration) — Understand config system
3. See Configuration section in [../README.md](../README.md) — Setup environment variables

### For Code Reviewers
1. Check [conventions/](conventions/) — Verify adherence to standards
2. Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#-layered-architecture) — Ensure proper layering
3. Look at [WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md) — Understand optimization logic

---

## 📊 Documentation Map

```
docs/
├── RESTRUCTURING_SUMMARY.md      ← START HERE
├── PROJECT_STRUCTURE.md          ← Architecture guide
├── WINDOW_AGGREGATION_OPTIMIZATION.md  ← Optimization details
├── IMPLEMENTATION_CHECKLIST.md   ← Rollout plan
├── conventions/
│   ├── 00-Index_and_glossary.md
│   ├── 01-Structure_conventions.md
│   ├── 02-Config_conventions.md
│   ├── 03-Naming_style_conventions.md
│   ├── 04-Dependencies_import_conventions.md
│   ├── 05-Error_handling_convention.md
│   ├── 06-Logging_observability_convention.md
│   ├── 07-Testing_convention.md
│   ├── 08-Security_secrets_conventions.md
│   ├── 09-Git_pr_release_convention.md
│   ├── 10-Code_design_principles.md
│   ├── 12-Api_conventions.md
│   ├── 13-data_ml_conventions.md
│   ├── 16-System_architecture_conventions.md
│   └── Example/
│       ├── config_example.py
│       └── ... (examples)
└── README.md (in root/../)  ← Getting started
```

---

## 🔍 Finding Information

### By Topic

**Architecture & Structure**
- Project layout: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Design patterns: [conventions/10-Code_design_principles.md](conventions/10-Code_design_principles.md)
- System design: [conventions/16-System_architecture_conventions.md](conventions/16-System_architecture_conventions.md)

**Configuration & Setup**
- Centralized config: [PROJECT_STRUCTURE.md#-centralized-configuration](PROJECT_STRUCTURE.md#-centralized-configuration)
- Config conventions: [conventions/02-Config_conventions.md](conventions/02-Config_conventions.md)
- Example configs: [conventions/Example/config_example.py](conventions/Example/config_example.py)

**Optimization & Performance**
- Window aggregation: [WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md)
- Performance metrics: [WINDOW_AGGREGATION_OPTIMIZATION.md#-performance-analysis](WINDOW_AGGREGATION_OPTIMIZATION.md#-performance-analysis)

**Development Practices**
- Naming standards: [conventions/03-Naming_style_conventions.md](conventions/03-Naming_style_conventions.md)
- Error handling: [conventions/05-Error_handling_convention.md](conventions/05-Error_handling_convention.md)
- Testing: [conventions/07-Testing_convention.md](conventions/07-Testing_convention.md)
- Logging: [conventions/06-Logging_observability_convention.md](conventions/06-Logging_observability_convention.md)

**Deployment & Operations**
- Rollout plan: [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)
- Infrastructure: [conventions/14-Infrastructure_deployment.md](conventions/14-Infrastructure_deployment.md)
- Git workflow: [conventions/09-Git_pr_release_convention.md](conventions/09-Git_pr_release_convention.md)

**Security & Best Practices**
- Security: [conventions/08-Security_secrets_conventions.md](conventions/08-Security_secrets_conventions.md)
- Code design: [conventions/10-Code_design_principles.md](conventions/10-Code_design_principles.md)
- Definition of done: [conventions/11-Definition_of_done.md](conventions/11-Definition_of_done.md)

---

## 💬 FAQ - Frequently Asked Questions

**Q: Where do I put new code?**  
A: See [PROJECT_STRUCTURE.md#-layered-architecture](PROJECT_STRUCTURE.md#-layered-architecture) to determine the right layer, then follow [conventions/01-Structure_conventions.md](conventions/01-Structure_conventions.md).

**Q: How do I configure the application?**  
A: Check [PROJECT_STRUCTURE.md#-centralized-configuration](PROJECT_STRUCTURE.md#-centralized-configuration) and [../config/app_config.py](../config/app_config.py).

**Q: What's the optimization about?**  
A: Quick version: [RESTRUCTURING_SUMMARY.md#3-window-aggregation-optimization](RESTRUCTURING_SUMMARY.md#3-window-aggregation-optimization)  
Detailed version: [WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md)

**Q: What's the migration path from old structure?**  
A: See [PROJECT_STRUCTURE.md#-migration-guide](PROJECT_STRUCTURE.md#-migration-guide).

**Q: How do I run the new code?**  
A: Check [../README.md](../README.md#-getting-started) for setup and usage.

**Q: Is the old code still supported?**  
A: Yes, for now. See [RESTRUCTURING_SUMMARY.md#-backward-compatibility](RESTRUCTURING_SUMMARY.md#-backward-compatibility).

---

## 📞 Getting Help

### For Questions About...

| Topic | Resource | Contact |
|-------|----------|---------|
| Project structure | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Document or team |
| Configuration | [config/app_config.py](../config/app_config.py) | Document or DevOps |
| Optimization | [WINDOW_AGGREGATION_OPTIMIZATION.md](WINDOW_AGGREGATION_OPTIMIZATION.md) | Document or Developer |
| Coding standards | [conventions/](conventions/) | Document or Tech Lead |
| Deployment | [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) | Document or DevOps |

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-10 | Initial restructuring, optimization module, comprehensive documentation |

---

## ✅ Quick Navigation

- 🏠 [Main README](../README.md) — Getting started (back to root)
- 📋 [Configuration](../config/app_config.py) — App config system
- 🔧 [Source Code](../src/) — Production code
- 🧪 [Tests](../tests/) — Test suite
- 📚 [Root Conventions](conventions/) — Coding standards

---

**Last Updated**: 2026-04-10  
**Maintained by**: Development Team  
**Status**: ✅ Documentation Complete
