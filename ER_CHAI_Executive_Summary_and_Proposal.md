# ER-CHAI: AI-Assisted Chart Generation & Analysis Platform
## Executive Summary and Proposal for Productionalization

**Prepared for:** Economic Research Leadership  
**Date:** April 2, 2026  
**Version:** 1.0  
**Classification:** Internal Use Only

---

## 1. Executive Summary

The Economic Research Chart and AI Helper (ER-CHAI) is a working prototype that enables economists and researchers to create, customize, and export publication-quality FRBSF-branded economic charts using AI-powered natural language commands. Built on a FastAPI/React stack with AWS Bedrock (Claude Sonnet 4.5), the tool currently runs as a single-user local application.

This proposal requests funding and resources to productionalize ER-CHAI into a secure, multi-user, cloud-hosted platform on AWS — making it available to the entire Economic Research team while adding significant AI-driven enhancements for chart generation and data analysis.

### What Exists Today (Prototype)

- FRED API data ingestion and CSV/Excel file upload
- Interactive Konva.js canvas editor with drag-and-drop chart elements
- AI assistant (Claude Sonnet 4.5) for natural language chart modification and data Q&A
- Reference image analysis via OpenCV + Bedrock Vision (replicates chart styling from screenshots)
- Auto-generated executive summaries with trend analysis and economist perspective
- Export to Python (matplotlib), R (ggplot2), and branded PDF
- Project save/load via local SQLite
- 211 unit and property-based tests

### What We're Proposing

Transform this prototype into a production-grade, multi-user AWS-hosted platform with enhanced AI capabilities, team collaboration features, and enterprise-grade security — enabling the Economic Research team to produce publication-ready charts 3-5x faster than current manual workflows.

---

## 2. Current Architecture Assessment

| Component | Current State | Production Gap |
|-----------|--------------|----------------|
| Backend | FastAPI, single-process, local | Needs containerization, auto-scaling, load balancing |
| Frontend | Vite dev server, localhost | Needs CDN hosting, build optimization |
| Database | SQLite (single-file) | Needs PostgreSQL/Aurora for multi-user concurrency |
| Authentication | None | Needs SSO/SAML integration with Okta IdP |
| AI/LLM | Direct Bedrock calls, no guardrails | Needs prompt caching, rate limiting, content filtering |
| Storage | Local filesystem | Needs S3 for datasets, charts, exports |
| Image Analysis | OpenCV + Bedrock Vision, in-process | Needs async processing queue for large images |
| Monitoring | Console logging | Needs CloudWatch, X-Ray, structured logging |
| Security | Local config.yaml with credentials | Needs Secrets Manager, IAM roles, VPC isolation |
| CI/CD | None | Needs automated pipeline with testing gates |

---

## 3. Proposed Production Architecture (AWS)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Cloud (VPC)                              │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │CloudFront│───▶│  ALB + WAF   │───▶│  ECS Fargate Cluster     │  │
│  │  (CDN)   │    │              │    │  ┌────────┐ ┌────────┐   │  │
│  └──────────┘    └──────────────┘    │  │Backend │ │Backend │   │  │
│       │                              │  │Task x2 │ │Task xN │   │  │
│  ┌──────────┐                        │  └────────┘ └────────┘   │  │
│  │ S3 Bucket│                        └──────────────────────────┘  │
│  │(Frontend)│                                    │                  │
│  └──────────┘                                    ▼                  │
│                              ┌─────────────────────────────────┐   │
│                              │         Service Layer           │   │
│                              │                                 │   │
│  ┌──────────────┐           │  ┌───────┐  ┌────────────────┐  │   │
│  │  Okta (IdP)  │◀──────────│  │Bedrock│  │  SQS Queue     │  │   │
│  │  SAML/OIDC   │           │  │Claude │  │(Async Analysis)│  │   │
│  └──────────────┘           │  └───────┘  └────────────────┘  │   │
│                              │                                 │   │
│  ┌──────────────┐           │  ┌───────┐  ┌────────────────┐  │   │
│  │ Aurora        │◀──────────│  │  S3   │  │  ElastiCache   │  │   │
│  │ PostgreSQL    │           │  │(Data) │  │  (Sessions)    │  │   │
│  └──────────────┘           │  └───────┘  └────────────────┘  │   │
│                              └─────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Observability: CloudWatch Logs + Metrics | X-Ray Tracing   │  │
│  │  Security: Secrets Manager | KMS | VPC Endpoints | GuardDuty│  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key AWS Services

| Service | Purpose | Justification |
|---------|---------|---------------|
| ECS Fargate | Backend compute | Serverless containers, auto-scaling, no server management |
| Aurora PostgreSQL Serverless v2 | Database | Multi-user concurrency, auto-scaling, point-in-time recovery |
| S3 | Static frontend hosting + dataset/export storage | Durable, cost-effective, versioned storage |
| CloudFront | CDN for frontend + API caching | Low-latency global delivery, DDoS protection |
| Okta (IdP) | Authentication & SSO | Organizational SAML/OIDC identity provider, MFA, RBAC |
| ALB OIDC Integration | Auth enforcement | ALB natively validates Okta JWT tokens at the edge |
| Bedrock | AI/LLM (Claude Sonnet) | Managed LLM access, no model hosting overhead |
| ElastiCache (Redis) | Session store + prompt caching | Fast AI session management, reduce Bedrock costs |
| SQS | Async job queue | Image analysis, PDF generation, bulk exports |
| Secrets Manager | Credential management | Rotate API keys, no hardcoded secrets |
| WAF | Web application firewall | OWASP protection, rate limiting |
| CloudWatch + X-Ray | Monitoring and tracing | Full observability, alerting, performance tracking |
| ECR | Container registry | Store and version Docker images |
| CodePipeline + CodeBuild | CI/CD | Automated build, test, deploy pipeline |

---

## 4. Proposed AI Enhancements

### Phase 1: Enhanced Chart Intelligence (Months 1-3)

| Enhancement | Description | Value |
|-------------|-------------|-------|
| Multi-series auto-detection | AI automatically identifies and suggests optimal chart types for multi-column datasets | Reduces manual configuration by 60% |
| Smart annotation suggestions | AI proactively suggests recession bands, policy event markers, and trend annotations based on data patterns | Adds analytical depth without manual research |
| Comparative chart generation | "Compare PCE vs CPI over the last 10 years" generates overlay charts from multiple FRED series | Enables rapid cross-indicator analysis |
| Natural language data filtering | "Show only data from 2020 onwards" or "Exclude outliers above 3 standard deviations" | Faster data exploration |

### Phase 2: Advanced AI Analysis (Months 3-5)

| Enhancement | Description | Value |
|-------------|-------------|-------|
| Automated anomaly detection | AI flags unusual data points, structural breaks, and regime changes | Early identification of economic shifts |
| Forecast overlay | AI generates and overlays simple forecasts (ARIMA, trend extrapolation) on charts | Forward-looking analysis without separate tools |
| Cross-dataset correlation | "What's the correlation between unemployment and inflation?" with visual output | Rapid hypothesis testing |
| Publication-ready narrative | AI generates full chart descriptions suitable for Economic Letters and working papers | Reduces writing time for publications |
| Template library | AI-curated chart templates for common economic indicators (PCE, GDP, unemployment, etc.) | Standardized, consistent output |

### Phase 3: Collaborative AI Features (Months 5-7)

| Enhancement | Description | Value |
|-------------|-------------|-------|
| Shared chart workspaces | Multiple researchers collaborate on the same chart in real-time | Team productivity |
| AI review assistant | AI reviews charts for branding compliance, data accuracy, and presentation quality | Quality assurance automation |
| Version history with AI diff | AI explains what changed between chart versions in plain language | Audit trail and context |
| Batch chart generation | "Generate quarterly PCE charts for all 12 Federal Reserve districts" | Bulk production capability |
| Custom AI personas | Configure AI behavior for different publication types (Economic Letter vs. Working Paper vs. Blog) | Tailored output quality |

---

## 5. Implementation Roadmap

### Phase 1: Foundation & Cloud Migration (Months 1-3)
- Containerize backend with Docker
- Migrate SQLite → Aurora PostgreSQL Serverless v2
- Deploy to ECS Fargate with ALB
- Host frontend on S3 + CloudFront
- Implement Okta SSO authentication (SAML/OIDC via ALB integration)
- Set up CI/CD pipeline (CodePipeline + CodeBuild)
- Migrate file storage to S3
- Implement CloudWatch logging and X-Ray tracing
- Deploy Phase 1 AI enhancements (smart chart types, annotation suggestions)
- Security hardening (WAF, VPC, Secrets Manager, KMS)

### Phase 2: AI Enhancements & Scaling (Months 3-5)
- Implement async processing queue (SQS) for image analysis and exports
- Add ElastiCache for AI session management and prompt caching
- Deploy Phase 2 AI features (anomaly detection, forecasting, correlation)
- Add FRED bulk data pipeline for pre-cached popular series
- Performance optimization and load testing
- User acceptance testing with research team

### Phase 3: Collaboration & Polish (Months 5-7)
- Real-time collaboration (WebSocket via API Gateway)
- Shared workspaces and team projects
- AI review assistant and batch generation
- Template library with curated economic chart templates
- Documentation, training materials, and onboarding guides
- Production launch and team rollout

---

## 6. Budget Estimate

### One-Time Development Costs (MVP — 2-3 Resources)

| Item | Effort | Rate | Cost |
|------|--------|------|------|
| Full-Stack Lead Engineer (backend, infra, AI) | 1 engineer × 7 months | $16,000/mo | $112,000 |
| Frontend / DevOps Engineer | 1 engineer × 6 months | $15,000/mo | $90,000 |
| Part-Time QA + Security Review | 0.5 engineer × 4 months | $14,000/mo | $28,000 |
| Project management (shared/part-time) | 0.25 PM × 7 months | $13,000/mo | $22,750 |
| **Subtotal — Development** | | | **$252,750** |

### Monthly AWS Infrastructure Costs (Estimated Steady-State)

| Service | Configuration | Monthly Cost |
|---------|--------------|-------------|
| ECS Fargate | 2 tasks (2 vCPU, 4GB each), auto-scale to 4 | $150 |
| Aurora PostgreSQL Serverless v2 | 0.5-4 ACU, 20GB storage | $120 |
| S3 | 100GB storage + requests | $15 |
| CloudFront | 100GB transfer/month | $20 |
| Bedrock (Claude Sonnet 4.5) | ~50,000 requests/month (est. 15 users) | $800 |
| ElastiCache (Redis) | cache.t4g.micro | $25 |
| Okta (IdP) | SSO license (assume org already has Okta) | $0* |
| CloudWatch + X-Ray | Logs, metrics, traces | $40 |
| Secrets Manager | 5 secrets | $5 |
| WAF | Basic rules | $10 |
| SQS | Standard queue | $5 |
| ECR | Image storage | $5 |
| Miscellaneous (data transfer, KMS, etc.) | | $30 |
| **Subtotal — Monthly Infrastructure** | | **$1,230/mo** |
| **Annual Infrastructure** | | **$14,760/yr** |

*\* Assumes organization already has Okta licensing. If new Okta tenant is needed, add ~$2/user/month.*

### Ongoing Maintenance Costs (Annual)

| Item | Cost |
|------|------|
| 0.5 FTE platform engineer (maintenance, updates, on-call) | $90,000/yr |
| AWS infrastructure | $14,760/yr |
| Bedrock usage growth buffer (20% YoY) | $2,000/yr |
| Security patching and compliance reviews | $10,000/yr |
| **Subtotal — Annual Maintenance** | **$116,760/yr** |

### Total Budget Summary

| Category | Cost |
|----------|------|
| One-time development (7 months, 2-3 resources) | $252,750 |
| Year 1 infrastructure (post-launch, ~5 months) | $6,150 |
| Year 1 total | **$258,900** |
| Annual recurring (Year 2+) | **$116,760/yr** |

---

## 7. Resource Requirements

### Development Team (MVP — 2-3 Resources, 7-Month Engagement)

| Role | Count | Duration | Responsibilities |
|------|-------|----------|-----------------|
| Full-Stack Lead Engineer | 1 | 7 months | Architecture, backend migration (FastAPI → Aurora, S3, Okta), AI/Bedrock enhancements, prompt engineering, code review |
| Frontend / DevOps Engineer | 1 | 6 months | React optimization, CDN deployment, IaC (CDK/Terraform), CI/CD pipeline, monitoring, real-time collaboration |
| QA + Security Engineer (part-time) | 0.5 | 4 months | Test automation, UAT coordination, security review, performance testing |
| Project Manager (shared) | 0.25 | 7 months | Sprint planning, stakeholder communication, risk management |

### Post-Launch (Steady State)

| Role | Count | Responsibilities |
|------|-------|-----------------|
| Platform Engineer | 0.5 FTE | Maintenance, monitoring, incident response, minor enhancements |
| Product Owner (existing staff) | 0.1 FTE | Prioritize feature requests, user feedback |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bedrock API cost overruns from heavy AI usage | Medium | Medium | Implement prompt caching (ElastiCache), usage quotas per user, cost alerts |
| Bedrock model deprecation or API changes | Low | High | Abstract LLM calls behind service interface, support model swapping |
| User adoption resistance | Medium | Medium | Early involvement of research team in UAT, training sessions, champion users |
| Data sensitivity concerns with Bedrock | Low | High | All data stays within AWS account, no external model training, VPC endpoints |
| Scope creep on AI enhancements | Medium | Medium | Phased delivery with clear acceptance criteria per phase |
| SSO/IdP integration complexity | Medium | Low | Start Okta OIDC integration early, use ALB native OIDC support to minimize custom code |

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chart creation time | 3-5x faster than current manual process | Time-tracking comparison study |
| Active monthly users | 80% of Economic Research team within 3 months of launch | Okta login analytics |
| AI assistant usage | 70% of charts use at least one AI command | Backend telemetry |
| Export volume | 50+ publication-ready exports per month | S3/export endpoint metrics |
| System availability | 99.5% uptime | CloudWatch synthetic monitoring |
| User satisfaction | NPS > 40 | Quarterly survey |

---

## 10. Recommendation

We recommend proceeding with the full 3-phase implementation using a lean MVP team of 2-3 resources. The prototype has already demonstrated the core value proposition — AI-assisted chart generation significantly accelerates the workflow for economic researchers. The investment of ~$259K in Year 1 and ~$117K annually thereafter is modest relative to the productivity gains across the research team. The lean team approach reduces upfront cost by nearly 50% compared to a full-staffed build, while still delivering all three phases within the same 7-month timeline by leveraging full-stack engineers who can work across the entire stack.

The phased approach allows us to deliver production value in Month 3 (cloud-hosted with basic AI enhancements) while continuing to build advanced capabilities through Month 7. This de-risks the investment by providing early validation with real users.

**Requested Approval:**
- $258,900 for Year 1 (development + infrastructure)
- $116,760/yr recurring budget starting Year 2
- Staffing authorization for 2-3 resources over 7 months (MVP team)

---

*Prepared by the ER-CHAI Development Team*  
*For questions, contact the project technical lead.*
