# LinkedIn Content Guide Deep Integration

## 🎯 Overview

The system now uses **deep integration** with the LinkedIn Content Guide to generate structured, high-quality LinkedIn posts and articles following a proven framework.

---

## 📚 LinkedIn Content Guide Structure

### **System Components**

The LinkedIn Content Guide is a comprehensive spec-based system with:

1. **5 Post Types**
   - **Narrative**: Story first, then lesson
   - **Jolt**: Short, sharp, consequence-heavy
   - **Insight**: Deep explanation of a pattern
   - **Contrarian**: Challenges standard advice
   - **Teaching**: Shows how to do one thing better

2. **50 Skeletons** (Content Patterns)
   - S01: Hidden Cost
   - S02: Everyone Gets This Wrong
   - S03: Bought Lesson
   - S04: Daily Blind Spot
   - S05: One Moment That Changes Everything
   - S06: Why You Feel Stuck
   - S07: What You Think vs What Actually Happens
   - S08: Quiet Skill That Changes Outcomes
   - S09: Nobody Tells You This Early
   - S10: Stop Doing X If You Want Y
   - ... (40 more skeletons)

3. **7 Sections** (Standard Structure)
   - **Hook**: Attention-grabbing opening
   - **Context**: Scene setting
   - **Insight**: Core truth or lesson
   - **Story**: Specific example
   - **Consequence**: What happens if ignored
   - **Shift**: Actionable change
   - **Close**: Reflective question or line

4. **Templates** (Section-Specific Prompts)
   - Each post type has specific prompts for each section
   - Example for "Narrative" Hook: "One short line that exposes tension or confusion from the moment"
   - Example for "Jolt" Hook: "Write a punchy truth that directly names the painful problem"

5. **Input-Aware System**
   - Adapts based on: tone, persona, industry, geography, challenge
   - Agnostic rules for missing inputs
   - Fuzzy matching for best template selection

---

## 🔧 How Deep Integration Works

### **Before (Light Integration)**
```
Topic → Generic MIMIR rules → AI generates → Output
```

### **After (Deep Integration)**
```
Topic + Tone + Persona + Industry
    ↓
Query LinkedIn Content Guide (fuzzy match)
    ↓
Extract: Post Type + Skeleton + Template + Section Prompts
    ↓
Query MIMIR Quality Rules
    ↓
Combine: Structure (LinkedIn Guide) + Quality (MIMIR)
    ↓
Enhanced prompt with explicit structure
    ↓
AI generates following exact structure
    ↓
Structured, high-quality output
```

---

## 📝 Implementation Details

### **New Method in `engine_kb_helper.py`**

```python
def get_linkedin_content_structure(
    self, 
    content_type,  # "post" or "article"
    tone,          # "sharp", "reflective", "teaching", etc.
    persona,       # "CXO", "Founder", "AE", etc.
    industry,      # "B2B SaaS", "Healthcare", etc.
    topic,         # Main topic
    challenge      # Optional specific challenge
):
    """
    Query LinkedIn Content Guide for specific structure
    Uses fuzzy matching to find best template
    Returns: Post type, skeleton, template, section prompts
    """
```

### **Updated LinkedIn Post Generation**

**In `post_generator.py` → `generate_linkedin_post()`:**

1. **Query LinkedIn Content Guide**
   ```python
   linkedin_structure = self.engine_kb.get_linkedin_content_structure(
       content_type="post",
       tone=tone,
       persona=audience,
       industry=target_industry,
       topic=topic_title
   )
   ```

2. **Query MIMIR Quality Rules**
   ```python
   mimir_rules = self.engine_kb.get_social_rules(
       "LinkedIn Post", 
       topic_title, 
       audience, 
       tone
   )
   ```

3. **Combine in Prompt**
   ```python
   prompt = f"""
   LINKEDIN CONTENT GUIDE STRUCTURE (FOLLOW EXACTLY):
   {linkedin_structure}
   
   EXECUTION INSTRUCTIONS:
   1. SELECT post type (narrative/jolt/insight/contrarian/teaching)
   2. SELECT skeleton from 50 options
   3. FOLLOW template section prompts
   4. ADAPT for persona and industry
   5. MAINTAIN tone
   
   MIMIR QUALITY RULES:
   {mimir_rules}
   """
   ```

### **Updated LinkedIn Article Generation**

**In `post_generator.py` → `generate_linkedin_article()`:**

Same approach but with:
- `content_type="article"`
- Expanded section instructions for 500-600 words
- More detailed paragraph guidance

---

## 🎨 Example: How It Works in Practice

### **Input**
```python
topic = "AI rewires sales execution"
tone = "sharp"
persona = "CXO"
industry = "B2B SaaS"
```

### **Step 1: Query LinkedIn Content Guide**

System queries for:
```
LinkedIn Content Guide for post:
- Topic: AI rewires sales execution
- Tone: sharp
- Target Persona: CXO
- Industry: B2B SaaS

Include:
- Post type selection (narrative, jolt, insight, contrarian, teaching)
- Skeleton selection from 50 skeletons
- Template with section prompts
- Section-specific guidance
- Voice and formatting rules
```

### **Step 2: Fuzzy Match Returns**

```
Post Type: JOLT (sharp tone matches jolt type)
Skeleton: S02 - "Everyone Gets This Wrong"
Template: T_JOLT

Section Prompts:
- Hook: "Write a punchy truth that directly names the painful problem"
- Context: "Describe a very common scene where this problem appears"
- Insight: "Explain the hidden cause behind the problem in simple language"
- Story: "Give one quick example that readers can picture in their mind"
- Consequence: "State the real cost of ignoring this. Show emotional and practical damage"
- Shift: "Offer one firm corrective move the reader can start immediately"
- Close: "End with a short line or question that lets the reader sit with the discomfort"
```

### **Step 3: AI Generates Following Structure**

```
Founder-led sales breaks at scale.

Not because founders aren't good at sales.
Because they can't clone themselves.

The real problem? You're treating it like a process issue.
It's a behavior rewiring challenge.

I've seen teams hire 5 AEs, give them the "founder playbook," 
and watch conversion drop 40%.

The cost isn't just lost deals.
It's buyer confusion. Rep anxiety. Pipeline unpredictability.

The fix isn't more coaching.
It's systematic execution rewiring at the moment level.

Are you scaling a process or rewiring behavior?

#B2BSaaS #GTM #SalesExecution
```

---

## 📊 Structure Enforcement

### **Post Structure (200-300 words)**

```
Hook (1 line)
    ↓
Context (2-4 lines)
    ↓
Insight (2-3 lines)
    ↓
Story (2-3 lines)
    ↓
Consequence (2-3 lines)
    ↓
Shift (2-3 lines)
    ↓
Close (1-2 lines)
    ↓
Hashtags (5-7)
```

### **Article Structure (500-600 words)**

```
Hook (2-3 sentences)
    ↓
Context (3-4 paragraphs)
    ↓
Insight (2-3 paragraphs)
    ↓
Story (2-3 paragraphs)
    ↓
Consequence (2 paragraphs)
    ↓
Shift (2-3 paragraphs)
    ↓
Close (1-2 paragraphs)
    ↓
Hashtags (5-7)
```

---

## 🎯 Post Type Selection Logic

### **Narrative**
- **Use When**: Story-driven content, personal lessons
- **Tone**: Reflective, teaching
- **Example Topics**: "The moment I realized...", "How I learned..."

### **Jolt**
- **Use When**: Sharp, wake-up call content
- **Tone**: Sharp, direct
- **Example Topics**: "Everyone gets this wrong", "Stop doing X"

### **Insight**
- **Use When**: Pattern explanation, deep analysis
- **Tone**: Analytical, teaching
- **Example Topics**: "Why X happens", "The hidden pattern in..."

### **Contrarian**
- **Use When**: Challenging common beliefs
- **Tone**: Bold, contrarian
- **Example Topics**: "Everyone says X, but...", "The truth about..."

### **Teaching**
- **Use When**: Skill improvement, how-to
- **Tone**: Teaching, practical
- **Example Topics**: "How to improve X", "One move that changes..."

---

## 🔍 Fuzzy Matching Strategy

The system uses **semantic similarity** to match inputs to templates:

1. **Primary Match**: Tone + Persona + Topic
2. **Secondary Match**: Tone + Persona
3. **Fallback**: General guidelines for content type

**Example Matching:**
```
Input: tone="sharp", persona="CXO", topic="sales mistakes"
    ↓
Matches: Post Type "Jolt" (sharp tone)
         Skeleton "Everyone Gets This Wrong" (mistakes)
         Template T_JOLT
```

---

## 📈 Expected Improvements

### **Before LinkedIn Guide Integration**
- Generic LinkedIn posts
- Inconsistent structure
- Weak hooks
- Unclear CTAs
- Variable quality

### **After LinkedIn Guide Integration**
- ✅ Structured posts following proven framework
- ✅ Strong, attention-grabbing hooks
- ✅ Clear narrative flow (hook → context → insight → story → consequence → shift → close)
- ✅ Persona-adapted language
- ✅ Industry-specific examples
- ✅ Tone-consistent throughout
- ✅ Reflective, engaging closes

---

## 🛠️ Usage

### **Generate LinkedIn Post**

```python
from post_generator import ContentPipeline

pipeline = ContentPipeline()

# System automatically:
# 1. Queries LinkedIn Content Guide
# 2. Selects post type, skeleton, template
# 3. Applies structure
# 4. Generates content

result = pipeline.generate_linkedin_post(
    topic={"title": "AI rewires sales execution"},
    related_news=[...],
    niche={...},
    audience="CXOs",
    tone="sharp",
    pdf_context="...",
    industry="B2B SaaS"
)
```

### **Generate LinkedIn Article**

```python
result = pipeline.generate_linkedin_article(
    topic={"title": "AI rewires sales execution"},
    related_news=[...],
    niche={...},
    audience="CXOs",
    tone="professional",
    pdf_context="...",
    industry="B2B SaaS"
)
```

---

## 🔍 Console Output

### **Successful Integration**

```
============================================================
🎯 DEEP LINKEDIN INTEGRATION
============================================================

🔍 LinkedIn Content Guide Query:
   Content Type: post
   Tone: sharp
   Persona: CXOs
   Industry: B2B SaaS
   Topic: AI rewires sales execution...
   Fetching structure...

📚 Retrieved 8 LinkedIn Guide chunks
   1. Linkedin Content Guide.pdf
      Preview: Post Type: JOLT - Short, sharp, consequence heavy...
   2. Linkedin Content Guide.pdf
      Preview: Skeleton S02: Everyone Gets This Wrong...
   3. Linkedin Content Guide.pdf
      Preview: Template T_JOLT section prompts...
   ...

✅ LinkedIn Content Guide structure: 4521 characters

============================================================
```

---

## 📚 Voice and Formatting Rules

From the LinkedIn Content Guide:

### **Voice**
- Sound like a human talking to one reader
- Use clear, simple language
- Avoid clichés, buzzwords, and jargon
- Focus on clarity and consequence

### **Formatting**
- No calls to action for likes or follows
- Close with a reflective line or question
- Do not use long dash character
- Use normal punctuation
- Short paragraphs for readability

### **Tone Adaptation**
- **Sharp**: Direct, punchy, consequence-focused
- **Reflective**: Thoughtful, introspective, lesson-focused
- **Teaching**: Clear, practical, step-by-step
- **Contrarian**: Bold, challenging, belief-flipping
- **Neutral**: Balanced, professional, accessible

---

## 🎓 Best Practices

### **1. Match Tone to Content**
- Sharp tone → Jolt post type
- Reflective tone → Narrative post type
- Teaching tone → Teaching post type
- Contrarian tone → Contrarian post type

### **2. Select Appropriate Skeleton**
- "Everyone Gets This Wrong" → Common mistakes
- "Hidden Cost" → Unseen losses
- "One Moment That Changes Everything" → Turning points
- "Why You Feel Stuck" → Blockages

### **3. Adapt for Persona**
- CXOs → Strategic, high-level, business impact
- Founders → Practical, growth-focused, tactical
- AEs → Sales-specific, deal-focused, execution
- Marketers → Campaign-focused, creative, data-driven

### **4. Industry-Specific Examples**
- B2B SaaS → Pipeline, conversion, GTM
- Healthcare → Patient care, compliance, innovation
- Finance → Risk, compliance, efficiency
- Retail → Customer experience, operations

---

## 🔄 Maintenance

### **Updating LinkedIn Content Guide**

1. Edit `Linkedin Content Guide.pdf` in `Engine rules/01-Content-Generation/`
2. Rebuild Engine KB:
   ```bash
   python engine_kb_builder.py
   ```
3. Test integration:
   ```bash
   python test_mimir_integration.py
   ```

### **Adding New Skeletons**

Add to the PDF, then rebuild Engine KB. The fuzzy matching will automatically include new skeletons.

---

## 📞 Troubleshooting

### **Problem: Generic posts still being generated**

**Check:**
1. Verify Engine KB was built: `ls -la ./engine_kb/vectordb/`
2. Check console for LinkedIn Guide query messages
3. Verify LinkedIn Content Guide PDF exists

**Solution:**
```bash
python engine_kb_builder.py
```

### **Problem: Structure not being followed**

**Cause:** AI not interpreting structure correctly

**Solution:** The prompt explicitly instructs to follow structure. If issues persist, increase temperature or try different model.

### **Problem: Wrong post type selected**

**Cause:** Fuzzy matching selected different type

**Solution:** Be more explicit with tone parameter:
- "sharp" → Jolt
- "reflective" → Narrative
- "teaching" → Teaching
- "contrarian" → Contrarian

---

## 📊 Quality Metrics

Track these to measure LinkedIn Guide integration success:

- **Structure Adherence**: Does output follow 7-section structure?
- **Hook Quality**: Is first line attention-grabbing?
- **Narrative Flow**: Does it flow hook → context → insight → story → consequence → shift → close?
- **Tone Consistency**: Is tone maintained throughout?
- **Persona Alignment**: Does language match target persona?
- **Engagement**: LinkedIn metrics (likes, comments, shares)

---

**Version:** 1.0  
**Last Updated:** November 14, 2025  
**Status:** ✅ Production Ready  
**Integration Type:** Deep (Structure Enforcement)
