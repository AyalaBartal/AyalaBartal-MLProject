# AI Tooling Usage

## Tools Used

This project used three AI tools throughout development:

### 1. Kiro (AWS AI IDE)
- **Usage**:
  - Built the initial project structure and Flask web application
  - Created the preprocessing pipeline and feature transformers
  - Generated unit and integration tests
  - Developed the Decision Tree and Random Forest training scripts
  - Set up the CI/CD GitHub Actions workflow

### 2. ChatGPT (OpenAI)
- **Usage**:
  - Helped debug preprocessing issues and feature alignment problems
  - Explained machine learning concepts and helped choose hyperparameters
  - Assisted with writing documentation and design explanations
  - Helped troubleshoot Render deployment issues

### 3. GitHub Copilot (Claude-based)
- **Usage**:
  - Fixed submission gaps — updated `evaluation-and-design.md` with real metrics from all 7 models
  - Computed missing CV metrics (Decision Tree 10-fold, Random Forest 10-fold with std, CatBoost test set)
  - Implemented functional `train.py` and `eval.py` scripts
  - Fixed CI/CD pipeline deploy step with real Render deploy hook
  - Fixed `.gitignore` so required submission docs are tracked in git
  - Added "Actual" and "Correct/Wrong" columns to batch results table
  - Fixed integration test to read feature count dynamically from schema

## What Worked Well
- **Rapid development**: Kiro generated large amounts of boilerplate code quickly — Flask routes, test suites, HTML templates
- **Debugging**: ChatGPT was useful for explaining errors and suggesting fixes
- **Gap filling**: GitHub Copilot was effective at identifying what was missing and making precise surgical fixes
- **Documentation**: All three tools helped write clear, professional documentation

## Challenges
- **Feature schema mismatches**: Different training scripts saved schemas with different key names (`feature_order` vs `features`) — required manual debugging
- **Dependency conflicts**: Python version and package compatibility issues on Render required manual adjustment
- **Duplicate columns**: The preprocessing pipeline produced duplicate column names that had to be deduplicated before training
- **Model accuracy too high**: Random Forest achieved near-perfect accuracy (AUC=1.0) making it hard to demonstrate errors in the demo — resolved by using Logistic Regression for comparison

## Effectiveness Assessment
- **Kiro**: ~80% of code generated — highly effective for structure and boilerplate
- **ChatGPT**: Useful for explanations and debugging guidance
- **GitHub Copilot**: Highly effective for precise fixes, metric computation, and submission preparation

**Moderately Effective (50-70% time savings)**:
- Debugging deployment issues
- Platform-specific configurations
- Performance optimization

**Manual Work Required**:
- Domain-specific feature naming
- Final testing and validation
- Deployment platform selection and setup

## Recommendations
AI tools are highly effective for:
- **Rapid application development** - Complete web apps in hours vs days
- **Comprehensive testing** - Automated generation of thorough test suites
- **Modern UI design** - Professional interfaces without design expertise
- **Documentation** - Complete, well-structured project documentation
- **CI/CD setup** - Industry-standard pipeline configurations

Manual oversight remains essential for:
- **Domain expertise** - Understanding malware analysis requirements
- **Platform compatibility** - Deployment-specific configurations
- **Final quality assurance** - End-to-end testing and validation
- **Performance optimization** - Production-ready fine-tuning
