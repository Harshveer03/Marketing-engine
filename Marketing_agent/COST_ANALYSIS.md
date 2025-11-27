# Marketing Engine - Cost Analysis & Pricing Strategy

**Document Version:** 1.0  
**Last Updated:** November 13, 2025  
**Analysis Period:** Monthly costs for SaaS deployment

---

## Executive Summary

This document provides a comprehensive cost analysis for deploying the Marketing Engine as a multi-tenant SaaS platform with authentication, storage, and API infrastructure.

**Key Findings:**

- **Cost per user (free tier):** $0.30 - $0.70/month
- **Recommended starting budget:** $150-200/month (100-200 beta users)
- **Break-even point:** ~40 paid users at $29/month
- **Primary cost driver:** AI API usage (60-70% of total costs)

---

## 1. AI API Costs

### 1.1 Google Gemini API (Primary Content Generation)

**Current Model:** `gemini-2.0-flash`

**Pricing:**

- Input tokens: $0.075 per 1M tokens
- Output tokens: $0.30 per 1M tokens

**Usage Estimates per User/Month:**

| Activity                   | Volume           | Tokens (Output) | Cost       |
| -------------------------- | ---------------- | --------------- | ---------- |
| Blog generation (10 posts) | 10 × 2,000 words | ~40,000         | $0.012     |
| Social posts (40 posts)    | 40 × 200 words   | ~30,000         | $0.009     |
| Topic generation           | 20 requests      | ~10,000         | $0.003     |
| Trend analysis             | 10 requests      | ~10,000         | $0.003     |
| **Total Output**           |                  | **~90,000**     | **$0.027** |
| Input tokens (prompts)     |                  | ~30,000         | $0.002     |
| **Grand Total**            |                  |                 | **$0.029** |

**User Segments:**

- **Light users** (5 blogs, 10 social): $0.04/month
- **Average users** (10 blogs, 40 social): $0.15/month
- **Heavy users** (30 blogs, 100 social): $0.50/month
- **Power users** (unlimited): $2.00+/month

### 1.2 SerpAPI (Trend Fetching)

**Pricing Tiers:**

- Free: 100 searches/month (not viable for production)
- Starter: $50/month for 5,000 searches
- Pro: $150/month for 15,000 searches

**Usage per User:**

- Average: 10 trend fetches/month
- Cost per search: $0.01 (on Starter plan)
- **Cost per user: $0.10/month**

**Optimization:**

- Cache trend results for 24 hours
- Reduces effective cost to: **$0.03/month per user**

### 1.3 Embeddings (Ollama vs OpenAI)

**Option A: Self-Hosted Ollama**

- Model: `nomic-embed-text`
- Cost: Server resources only
- **Per user: $0.00** (absorbed in infrastructure)

**Option B: OpenAI Embeddings**

- Model: `text-embedding-3-small`
- Pricing: $0.02 per 1M tokens
- Usage: ~5,000 tokens per user setup
- **Per user: $0.0001** (negligible)

**Recommendation:** Self-host Ollama for cost efficiency

---

## 2. Infrastructure Costs

### 2.1 Application Hosting

**Platform Options:**

| Provider     | Tier          | Specs             | Users Supported | Monthly Cost |
| ------------ | ------------- | ----------------- | --------------- | ------------ |
| Railway      | Starter       | 512MB RAM, 1 vCPU | ~100            | $5           |
| Railway      | Pro           | 2GB RAM, 2 vCPU   | ~500            | $20          |
| DigitalOcean | Basic Droplet | 2GB RAM, 1 vCPU   | ~200            | $12          |
| DigitalOcean | Standard      | 4GB RAM, 2 vCPU   | ~1,000          | $24          |
| AWS EC2      | t3.small      | 2GB RAM, 2 vCPU   | ~300            | $15          |
| AWS EC2      | t3.medium     | 4GB RAM, 2 vCPU   | ~1,000          | $30          |

**Recommended:** Railway Pro or DigitalOcean Standard

**Per User Cost:** $0.04 - $0.10/month (at scale)

### 2.2 Database (PostgreSQL)

**Requirements:**

- User accounts & authentication
- Content metadata
- Usage tracking
- Analytics data

**Platform Options:**

| Provider     | Tier        | Storage | Connections | Monthly Cost |
| ------------ | ----------- | ------- | ----------- | ------------ |
| Supabase     | Free        | 500MB   | 50          | $0           |
| Supabase     | Pro         | 8GB     | 200         | $25          |
| Railway      | Postgres    | 5GB     | 100         | $10          |
| DigitalOcean | Managed DB  | 10GB    | 25          | $15          |
| AWS RDS      | db.t3.micro | 20GB    | 100         | $15          |

**Recommended:** Start with Supabase Free, upgrade to Pro at 200+ users

**Per User Cost:** $0.05 - $0.10/month

### 2.3 File Storage (User Documents & Generated Content)

**Storage Requirements per User:**

- Uploaded documents: ~5-10MB
- Generated content (JSON): ~5MB
- Vector embeddings: ~20MB
- Total: ~30-35MB per user

**Platform Options:**

| Provider            | Pricing Model              | Cost per GB | 100 Users Cost | 1,000 Users Cost |
| ------------------- | -------------------------- | ----------- | -------------- | ---------------- |
| AWS S3              | $0.023/GB                  | $0.023      | $0.08          | $0.80            |
| Cloudflare R2       | $0.015/GB                  | $0.015      | $0.05          | $0.50            |
| Supabase Storage    | Free (1GB), then $0.021/GB | $0.021      | Included       | $0.70            |
| DigitalOcean Spaces | $5/month (250GB)           | Flat        | $5             | $5               |

**Recommended:** DigitalOcean Spaces (predictable flat rate)

**Per User Cost:** $0.01 - $0.02/month

### 2.4 Vector Database (FAISS Storage)

**Current Implementation:** File-based FAISS (stored with user data)

**Scaling Options:**

| Solution         | Type        | Pricing                  | Per User Cost |
| ---------------- | ----------- | ------------------------ | ------------- |
| File-based FAISS | Self-hosted | Included in storage      | $0.00         |
| Pinecone         | Managed     | $70/month (100K vectors) | $0.10         |
| Weaviate Cloud   | Managed     | $25/month (starter)      | $0.05         |
| Qdrant Cloud     | Managed     | $49/month (1M vectors)   | $0.05         |

**Recommendation:** Keep file-based FAISS until 500+ users

**Per User Cost:** $0.00 - $0.05/month

---

## 3. Authentication & User Management

### 3.1 Auth Service Options

| Provider             | Free Tier  | Paid Tier             | Features                  | Recommendation   |
| -------------------- | ---------- | --------------------- | ------------------------- | ---------------- |
| Supabase Auth        | 50,000 MAU | $25/month (100K MAU)  | Email, OAuth, Magic Links | ⭐ Best choice   |
| Auth0                | 7,000 MAU  | $35/month (1,000 MAU) | Enterprise features       | Overkill         |
| Firebase Auth        | 50,000 MAU | Pay-as-you-go         | Google integration        | Good alternative |
| Custom (Flask-Login) | Free       | Free                  | Full control              | More dev work    |

**Recommended:** Supabase Auth (free tier sufficient for launch)

**Per User Cost:** $0.00 (free tier), $0.25/month (paid tier)

### 3.2 Session Management

**Redis for Session Storage:**

- Railway Redis: $5/month (512MB)
- Upstash Redis: Free tier (10K requests/day)
- **Per User Cost:** $0.01/month

---

## 4. Communication Services

### 4.1 Email Service (Transactional)

**Use Cases:**

- Email verification
- Password reset
- Usage notifications
- Marketing emails (optional)

**Provider Options:**

| Provider | Free Tier           | Paid Pricing              | Per User Cost |
| -------- | ------------------- | ------------------------- | ------------- |
| SendGrid | 100 emails/day      | $19.95/month (50K emails) | $0.01         |
| Mailgun  | 5,000 emails/month  | $35/month (50K emails)    | $0.01         |
| AWS SES  | 62,000 emails/month | $0.10 per 1,000           | $0.001        |
| Resend   | 3,000 emails/month  | $20/month (50K emails)    | $0.01         |

**Recommended:** AWS SES (most cost-effective at scale)

**Per User Cost:** $0.001 - $0.01/month

---

## 5. Monitoring & Analytics

### 5.1 Application Monitoring

**Options:**

- **Sentry** (Error tracking): Free tier (5K events/month)
- **LogRocket** (Session replay): $99/month (1K sessions)
- **Datadog** (APM): $15/host/month

**Recommended:** Sentry free tier initially

**Cost:** $0 - $15/month (fixed)

### 5.2 Usage Analytics

**Options:**

- **PostHog** (Product analytics): Free tier (1M events)
- **Mixpanel** (User analytics): Free tier (100K events)
- **Custom** (PostgreSQL queries): Free

**Recommended:** PostHog free tier

**Cost:** $0 (free tier sufficient)

---

## 6. Total Cost Summary

### 6.1 Cost Per User (Monthly)

| Component          | Light User | Average User | Heavy User | Power User |
| ------------------ | ---------- | ------------ | ---------- | ---------- |
| **AI APIs**        |
| Gemini API         | $0.04      | $0.15        | $0.50      | $2.00      |
| SerpAPI (cached)   | $0.02      | $0.03        | $0.05      | $0.10      |
| Embeddings         | $0.00      | $0.00        | $0.00      | $0.00      |
| **Infrastructure** |
| Hosting            | $0.05      | $0.08        | $0.10      | $0.15      |
| Database           | $0.03      | $0.05        | $0.08      | $0.10      |
| Storage            | $0.01      | $0.01        | $0.02      | $0.03      |
| Vector DB          | $0.00      | $0.00        | $0.02      | $0.05      |
| **Services**       |
| Auth               | $0.00      | $0.00        | $0.00      | $0.00      |
| Email              | $0.01      | $0.01        | $0.01      | $0.02      |
| **TOTAL**          | **$0.16**  | **$0.33**    | **$0.78**  | **$2.45**  |

### 6.2 Fixed Monthly Costs

| Service             | Cost    | Notes                          |
| ------------------- | ------- | ------------------------------ |
| Application Hosting | $20     | Railway Pro / DO Standard      |
| Database            | $25     | Supabase Pro (after free tier) |
| File Storage        | $5      | DigitalOcean Spaces            |
| Redis               | $5      | Railway Redis                  |
| Monitoring          | $0      | Free tiers                     |
| **Total Fixed**     | **$55** | **Base infrastructure**        |

---

## 7. Scaling Scenarios

### 7.1 Beta Launch (100 Users)

**User Distribution:**

- 60% Light users (60)
- 30% Average users (30)
- 10% Heavy users (10)

**Costs:**

- Variable: (60 × $0.16) + (30 × $0.33) + (10 × $0.78) = $27.30
- Fixed: $55
- **Total: $82.30/month**

### 7.2 Growth Phase (500 Users)

**User Distribution:**

- 50% Light users (250)
- 35% Average users (175)
- 15% Heavy users (75)

**Costs:**

- Variable: (250 × $0.16) + (175 × $0.33) + (75 × $0.78) = $156.25
- Fixed: $80 (upgraded infrastructure)
- **Total: $236.25/month**

### 7.3 Scale Phase (1,000 Users)

**User Distribution:**

- 50% Light users (500)
- 30% Average users (300)
- 15% Heavy users (150)
- 5% Power users (50)

**Costs:**

- Variable: (500 × $0.16) + (300 × $0.33) + (150 × $0.78) + (50 × $2.45) = $418
- Fixed: $150 (scaled infrastructure)
- **Total: $568/month**

---

## 8. Cost Optimization Strategies

### 8.1 Usage Limits (Recommended)

**Free Tier Limits:**

- 5 blogs per month
- 10 social posts per month
- 3 trend fetches per month
- 1 document upload

**Impact:**

- Reduces average user cost from $0.33 to $0.18
- **Savings: 45%**

### 8.2 Caching Strategy

**Implementation:**

- Cache trend results for 24 hours
- Cache topic generations for 1 hour
- Cache embeddings permanently

**Impact:**

- Reduces SerpAPI calls by 70%
- Reduces Gemini API calls by 20%
- **Savings: 25-30% on API costs**

### 8.3 Batch Processing

**Implementation:**

- Queue content generation requests
- Process in batches during off-peak hours
- Use spot instances for processing

**Impact:**

- Reduces server costs by 30%
- **Savings: $15-20/month at 500 users**

### 8.4 Content Lifecycle Management

**Implementation:**

- Auto-delete generated content after 90 days
- Compress old documents
- Archive inactive user data

**Impact:**

- Reduces storage costs by 60%
- **Savings: $5-10/month at 1,000 users**

### 8.5 API Rate Limiting

**Implementation:**

- 10 requests per minute per user
- 100 requests per day per user
- Exponential backoff for retries

**Impact:**

- Prevents API abuse
- Protects against cost spikes
- **Savings: Prevents 10-20% cost overruns**

---

## 9. Pricing Strategy

### 9.1 Recommended Pricing Tiers

| Tier         | Price     | Features                                                  | Target User                |
| ------------ | --------- | --------------------------------------------------------- | -------------------------- |
| **Free**     | $0        | 5 blogs/month, 10 social posts/month, 3 trend fetches     | Hobbyists, testers         |
| **Starter**  | $19/month | 20 blogs/month, 50 social posts/month, 10 trend fetches   | Solopreneurs, freelancers  |
| **Pro**      | $49/month | 100 blogs/month, 200 social posts/month, unlimited trends | Small businesses, agencies |
| **Business** | $99/month | Unlimited content, priority support, API access           | Agencies, enterprises      |

### 9.2 Break-Even Analysis

**Assumptions:**

- Average cost per user: $0.33/month
- Fixed costs: $55/month

**Break-Even Points:**

| Tier     | Monthly Price | Users Needed | MRR at Break-Even |
| -------- | ------------- | ------------ | ----------------- |
| Starter  | $19           | 3 paid users | $57               |
| Pro      | $49           | 2 paid users | $98               |
| Business | $99           | 1 paid user  | $99               |

**Realistic Scenario (100 total users):**

- 70 Free users: -$23 (70 × $0.33)
- 20 Starter users: +$380 (20 × $19)
- 8 Pro users: +$392 (8 × $49)
- 2 Business users: +$198 (2 × $99)
- Fixed costs: -$55
- **Net Profit: +$892/month**

### 9.3 Conversion Assumptions

**Industry Benchmarks:**

- Free to Paid conversion: 2-5%
- Starter to Pro upgrade: 20-30%
- Pro to Business upgrade: 10-15%

**Conservative Projections (1,000 users):**

- 970 Free users
- 20 Starter users ($380 MRR)
- 8 Pro users ($392 MRR)
- 2 Business users ($198 MRR)
- **Total MRR: $970**
- **Total Costs: $568**
- **Net Profit: $402/month**

---

## 10. Risk Mitigation

### 10.1 Cost Spike Prevention

**Measures:**

- Hard limits on API usage per user
- Rate limiting at application level
- Alert thresholds at 80% of budget
- Automatic user suspension at 150% of tier limit

### 10.2 Abuse Prevention

**Measures:**

- CAPTCHA on signup
- Email verification required
- Credit card required for free trial (no charge)
- Automated abuse detection (unusual patterns)

### 10.3 Budget Monitoring

**Tools:**

- AWS Cost Explorer
- Custom usage dashboard
- Weekly cost reports
- Per-user cost tracking

**Alert Thresholds:**

- Daily spend > $50
- Per-user cost > $5
- API error rate > 5%

---

## 11. Recommendations

### 11.1 Launch Strategy

**Phase 1: Beta (Month 1-2)**

- 100-200 invited users
- Free tier only
- Budget: $150-200/month
- Focus: Product validation, feedback

**Phase 2: Soft Launch (Month 3-4)**

- Open signups with waitlist
- Introduce Starter tier ($19/month)
- Target: 500 users, 20 paid
- Budget: $300/month
- Revenue: $380/month
- **Net: +$80/month**

**Phase 3: Public Launch (Month 5-6)**

- Full marketing push
- All tiers available
- Target: 1,000 users, 30 paid
- Budget: $600/month
- Revenue: $1,000/month
- **Net: +$400/month**

### 11.2 Critical Success Factors

1. **Keep free tier costs under $0.20/user**

   - Implement strict usage limits
   - Aggressive caching
   - Batch processing

2. **Achieve 3% free-to-paid conversion**

   - Clear value proposition
   - Friction-free upgrade path
   - Usage-based nudges

3. **Maintain 90%+ user satisfaction**
   - Fast content generation
   - High-quality output
   - Responsive support

### 11.3 Red Flags to Watch

- **Cost per user > $1.00**: Review API usage, implement limits
- **Conversion rate < 2%**: Improve onboarding, value communication
- **Churn rate > 10%**: Investigate product-market fit issues
- **API error rate > 3%**: Infrastructure scaling needed

---

## 12. Conclusion

**Key Takeaways:**

1. **Viable at small scale**: $150-200/month for 100-200 beta users
2. **Profitable at 30 paid users**: Break-even with conservative pricing
3. **Primary cost driver**: AI API usage (60-70% of variable costs)
4. **Optimization is critical**: Caching and limits reduce costs by 40-50%
5. **Freemium model works**: With proper limits and conversion funnel

**Next Steps:**

1. Implement usage limits and caching
2. Set up cost monitoring and alerts
3. Launch beta with 100 invited users
4. Validate pricing with early adopters
5. Iterate based on actual usage patterns

---

**Document Prepared By:** AI Assistant  
**For:** Marketing Engine SaaS Launch  
**Contact:** [Your contact information]
