# Executive Summary: ER-CHAI (Economic Research Chart AI)
## AI-Assisted Chart Generation & Analysis Platform - MVP Proposal

**Prepared for:** FRBSF Leadership  
**Date:** April 2026  
**Version:** 1.0 (MVP)

---

## 1. Executive Overview

The **ER-CHAI (Economic Research Chart AI)** platform is an innovative AI-powered tool designed to revolutionize how teams create, analyze, and publish economic charts. Currently in prototype phase, this proposal seeks funding to deliver an **MVP (Minimum Viable Product)** deployment on AWS within **3-4 months**.

### Vision Statement
*Empower users with AI-assisted chart generation that reduces manual effort by 70%, ensures brand consistency, and accelerates publication timelines from days to hours.*

---

## 2. Current State & Prototype Capabilities

### Existing Features (Prototype)
| Capability | Description |
|------------|-------------|
| **FRED API Integration** | Direct data ingestion from Federal Reserve Economic Data |
| **AI Chart Editor** | Natural language commands via Claude Sonnet 4.5 |
| **Reference Image Analysis** | OpenCV + Bedrock Vision extracts styling from existing charts |
| **Interactive Canvas** | Drag-and-drop Konva.js editor with real-time preview |
| **Multi-format Export** | Python, R, and PDF exports with FRBSF branding |
| **Executive Summary Generation** | AI-generated trend analysis and economist perspectives |
| **Project Management** | SQLite-based save/load functionality |

### Technical Stack
- **Backend:** Python 3.11+, FastAPI, uvicorn
- **Frontend:** React 18, TypeScript, Konva.js, Zustand
- **AI/ML:** AWS Bedrock (Claude Sonnet 4.5), OpenCV
- **Database:** SQLite (prototype) → PostgreSQL/RDS (production)
- **Testing:** 211 unit tests with pytest

---

## 3. Business Case & ROI Analysis

### Current Pain Points
1. **Manual Chart Creation:** 2-4 hours per chart for complex economic visualizations
2. **Inconsistent Branding:** Multiple style variations across publications
3. **Limited Collaboration:** No centralized platform for chart sharing
4. **Repetitive Tasks:** Reformatting data, applying annotations, generating summaries

### Projected Benefits

| Metric | Current State | With ER-CHAI | Improvement |
|--------|---------------|--------------|-------------|
| Chart Creation Time | 2-4 hours | 15-30 minutes | **75-85% reduction** |
| Brand Consistency | ~70% | 99%+ | **Near-perfect compliance** |
| Publication Cycle | 3-5 days | Same day | **80% faster** |
| Economist Productivity | Baseline | +40% capacity | **Significant uplift** |

### ROI Calculation (3-Year)
- **Estimated Time Savings:** 2,000+ hours/year across team
- **Cost Avoidance:** $150,000-$200,000/year in productivity gains
- **Total 3-Year ROI:** **250-300%** on initial investment

---

## 4. Proposed Enhancements

### MVP Phase (Q2-Q3 2026) - 3-4 Months
- [ ] **Okta SSO Integration** - Enterprise authentication via existing Okta IDP
- [ ] PostgreSQL/RDS migration for production scalability
- [ ] Basic role-based access control (RBAC)
- [ ] Essential audit logging
- [ ] Production deployment on AWS

### Future Enhancements (Post-MVP)
- [ ] Multi-chart dashboards
- [ ] Automated data refresh scheduling
- [ ] Advanced analytics and forecasting
- [ ] Real-time collaboration
- [ ] Template library

---

## 5. AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Route 53  │───▶│    ALB      │───▶│    ECS      │          │
│  │   (DNS)     │    │ (Load Bal.) │    │  (Fargate)  │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                               │                  │
│  ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐          │
│  │  Okta IDP   │    │     S3      │    │    RDS      │          │
│  │   (SSO)     │    │  (Assets)   │    │ (PostgreSQL)│          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Bedrock    │    │ CloudWatch  │    │   Secrets   │          │
│  │  (Claude)   │    │ (Logging)   │    │   Manager   │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Key AWS Services (MVP)
| Service | Purpose | Estimated Monthly Cost |
|---------|---------|------------------------|
| ECS Fargate | Container hosting | $150-250 |
| RDS PostgreSQL | Database (small instance) | $100-150 |
| ALB | Load balancing | $30-50 |
| S3 | Asset storage | $10-20 |
| Bedrock | AI/ML inference | $300-800 |
| Okta IDP | Authentication (existing) | $0 (existing license) |
| CloudWatch | Monitoring | $30-50 |
| **Total** | | **$620-1,320/month** |

---

## 6. Resource Requirements

### Team Composition (MVP)

| Role | FTE | Duration | Responsibility |
|------|-----|----------|----------------|
| **Tech Lead/Developer** | 1.0 | 4 months | Architecture, backend, AWS |
| **Full-Stack Developer** | 1.0 | 4 months | Frontend, Okta integration |
| **DevOps/QA** | 0.5 | 4 months | CI/CD, testing, deployment |

### MVP Timeline (3-4 Months)

```
Month 1: Foundation
    ├── AWS environment setup
    ├── Okta SSO integration
    └── Database migration to RDS

Month 2: Core Development
    ├── Production hardening
    ├── Security implementation
    └── Performance optimization

Month 3: Testing & Deployment
    ├── UAT with pilot users
    ├── Bug fixes and refinements
    └── Production deployment

Month 4: Stabilization (if needed)
    ├── Production monitoring
    ├── User training
    └── Documentation
```

---

## 7. Budget Request (MVP)

### MVP Development Costs

| Category | Amount |
|----------|--------|
| Development Team (2.5 FTE × 4 months) | $160,000 |
| AWS Infrastructure Setup | $10,000 |
| Security Review | $10,000 |
| Training & Documentation | $5,000 |
| Contingency (15%) | $27,750 |
| **Total MVP** | **$212,750** |

### Recurring Costs (Annual)

| Category | Amount |
|----------|--------|
| AWS Infrastructure | $7,500-16,000 |
| Bedrock AI Usage | $3,600-9,600 |
| Maintenance & Support (0.25 FTE) | $40,000 |
| **Total Annual** | **$51,100-65,600** |

### Total Budget Ask

| Period | Amount |
|--------|--------|
| **MVP Development (4 months)** | **$212,750** |
| **Year 1 Operations (8 months)** | **$34,000-44,000** |
| **Year 2 Operations** | **$51,100-65,600** |
| **Total (MVP + 20 months ops)** | **$297,850 - $322,350** |

---

## 8. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AWS Bedrock availability | Low | High | Multi-region failover, fallback models |
| Data security concerns | Medium | High | Encryption, VPC isolation, audit logs |
| User adoption resistance | Medium | Medium | Training, change management, champions |
| Scope creep | Medium | Medium | Agile methodology, prioritized backlog |
| Integration complexity | Low | Medium | Phased rollout, thorough testing |

---

## 9. Success Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chart creation time | <30 minutes avg | System analytics |
| User adoption rate | >80% of team | Monthly active users |
| Brand compliance | >95% | Automated style checks |
| System uptime | 99.5% | CloudWatch monitoring |
| User satisfaction | >4.2/5.0 | Quarterly surveys |
| AI accuracy | >90% | Command success rate |

---

## 10. Recommendation

We recommend **approval** of the **$213K MVP budget** to productionalize ER-CHAI within 3-4 months. The platform will:

1. **Dramatically improve productivity** across teams
2. **Ensure consistent FRBSF branding** across all publications
3. **Leverage cutting-edge AI** via AWS Bedrock
4. **Integrate with existing Okta SSO** for seamless authentication

### Next Steps
1. Secure MVP budget approval (~$213K)
2. Finalize AWS architecture review
3. Assign development team (2.5 FTE)
4. Kick off MVP development (May 2026)
5. Target production launch: **August 2026**

---

**Prepared by:** Economic Research Technology Team  
**Contact:** [Project Lead Email]  
**Attachments:** Detailed Technical Specification, AWS Architecture Diagram, Risk Register
