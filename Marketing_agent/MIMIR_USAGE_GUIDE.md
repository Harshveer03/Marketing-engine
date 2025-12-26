# MIMIR Integration Usage Guide

## 🎯 Overview

Your marketing engine now uses the **MIMIR 16-part content generation system** to create high-quality, persona-aligned content across all platforms.

---

## 🚀 Quick Start

### **1. Build the Engine KB (First Time Only)**

```bash
python engine_kb_builder.py
```

This processes all 20 MIMIR PDFs and creates a searchable vector database.

**Expected Output:**
```
🔧 Building Engine Knowledge Base...
📚 Found 20 PDF files
📄 Loading: MIMIR-01-Defined.pdf
   ✅ Loaded 15 pages
...
✅ Engine KB created successfully!
   Location: ./engine_kb/vectordb
   Total chunks: 847
   Source files: 20
```

### **2. Test the Integration**

```bash
python test_mimir_integration.py
```

**Expected Output:**
```
🧪 MIMIR INTEGRATION TEST SUITE
✅ PASS: Engine KB Availability
✅ PASS: Engine KB Helper
✅ PASS: Blog Generator Integration
✅ PASS: Social Generator Integration
✅ PASS: MIMIR Rule Query

5/5 tests passed
🎉 All tests passed! MIMIR integration is working correctly.
```

### **3. Generate Content**

```bash
# Web Dashboard
python app.py

# Or CLI
python marketing_engine.py
```

---

## 📊 How MIMIR Enhances Your Content

### **Before MIMIR:**
```
Generic prompt → AI generates content → Basic quality check
```

### **After MIMIR:**
```
Topic + Audience + Tone
    ↓
Query MIMIR Rules (16-part system)
    ↓
Enhanced prompt with specific guidelines
    ↓
AI generates content following MIMIR framework
    ↓
Quality validation against MIMIR criteria
```

---

## 🎨 MIMIR in Action

### **Example: LinkedIn Post Generation**

**Without MIMIR:**
```
"Create a LinkedIn post about AI in sales for CXOs"
```

**With MIMIR:**
```
Create a LinkedIn post about AI in sales for CXOs

MIMIR CONTENT GENERATION RULES:
- Intent & Grounding: Clarify message intent for CXO audience
- Tone Decision: Professional, authoritative, strategic
- Tailoring Principles: Adapt to B2B SaaS CXO persona
- Structural Tailoring: Optimize for LinkedIn's format
- Logic-Emotion Balance: Balance data with compelling narrative
- Anti-Patterns: Avoid clichés like "game-changer", "revolutionary"
- Integrity & Grounding: Fact-based, no hallucination
```

**Result:** More targeted, professional, and effective content.

---

## 📝 Platform-Specific MIMIR Application

### **LinkedIn Posts**
**MIMIR Parts Used:** 1, 4, 7, 8, 15

**Focus:**
- Professional tone calibration
- CXO-level language
- Platform-optimized structure
- Thought leadership positioning

**Example Output:**
```
The shift from founder-led to scalable GTM isn't just operational—
it's behavioral. 

Most B2B SaaS teams fail here because they treat it as a process 
problem when it's actually an execution rewiring challenge.

Here's what we're seeing work:
• Real-time rep fluency scoring
• Transcript-based rewrite drills
• Decision signal detection at scale

The companies winning aren't just coaching better—they're 
systematically rewiring how their teams execute in critical moments.

#B2BSaaS #GTM #SalesExecution #RevenueGrowth
```

### **LinkedIn Articles**
**MIMIR Parts Used:** 3, 5, 12, 13

**Focus:**
- Narrative structure and pacing
- Cliché removal
- Factual grounding
- Story coherence

**Example Structure:**
```
# AI Rewires Sales Execution: Drive Immediate Revenue Impact

## Introduction
[Hook with specific problem statement]

## The Challenge
[Data-backed analysis of current state]

## The Solution
[Concrete, actionable approach]

## Implementation
[Step-by-step guidance]

## Conclusion
[Forward-looking perspective]
```

### **Twitter/X**
**MIMIR Parts Used:** 1, 2, 8, 15

**Focus:**
- Concise phrasing
- Platform-specific format
- High engagement
- Punchy delivery

**Example Output:**
```
Founder-led sales breaks at scale.

Not because founders aren't good at sales.

Because they can't clone themselves.

The fix isn't hiring more reps.
It's rewiring how reps execute.

#B2BSaaS #GTM
```

### **YouTube**
**MIMIR Parts Used:** 3, 12, 14, 15

**Focus:**
- Video narrative structure
- Visual orchestration
- Story pacing
- Engagement hooks

**Example Script:**
```
[0:00-0:10] Hook
"Your sales team is ghosting buyers. Not intentionally. 
But it's happening. Here's why..."

[0:10-0:30] Problem Setup
"Most B2B SaaS companies hit a wall when founder-led sales 
needs to scale..."

[0:30-2:00] Core Content
[Structured explanation with visual cues]

[2:00-2:30] Solution
[Actionable framework]

[2:30-3:00] CTA
"If you're a GTM leader facing this challenge..."
```

### **Blogs**
**MIMIR Parts Used:** 1, 2, 3, 4, 7, 12, 13, 15

**Focus:**
- Comprehensive structure
- Authoritative tone
- Narrative flow
- Factual accuracy
- Persona alignment

**Example Structure:**
```
Title: AI Rewires Sales Execution: Drive Immediate Revenue Impact

[Introduction - 150 words]
- Hook with specific problem
- Establish authority
- Preview value

[Section 1: The Challenge - 200 words]
- Current state analysis
- Pain point exploration
- Industry context

[Section 2: The Solution - 250 words]
- Framework introduction
- Concrete examples
- Data points

[Section 3: Implementation - 200 words]
- Actionable steps
- Best practices
- Common pitfalls

[Conclusion - 100 words]
- Summary
- Forward perspective
- CTA
```

---

## 🔍 Monitoring MIMIR Integration

### **Console Output to Watch For:**

**Successful Integration:**
```
✅ Engine KB (MIMIR) loaded for content generation
🧠 Fetching MIMIR rules for LinkedIn Post...
✅ MIMIR rules loaded: 2847 chars
📝 LinkedIn Post: Generating content for topic: 'AI in Sales'
✅ LinkedIn Post content generated successfully
```

**Engine KB Not Available:**
```
⚠️ Engine KB not available: [Errno 2] No such file or directory: './engine_kb/vectordb'
⚠️ Could not load MIMIR rules: vectordb is None
```
**Action:** Run `python engine_kb_builder.py`

---

## 🎓 Understanding MIMIR Parts

### **The 16 Parts (Quick Reference)**

1. **Intent & Grounding** - Clarify message purpose
2. **Phrasing Foundations** - Sentence construction
3. **Structure Architecture** - Content organization
4. **Tone Decision** - Voice and style
5. **Anti-Patterns Removal** - Remove weak language
6. **Application Logic** - Technical accuracy
7. **Tailoring Principles** - Persona adaptation
8. **Structural Tailoring** - Platform optimization
9. **Cognitive Bias Counter** - Logical reasoning
10. **Evolution & Feedback** - Self-correction
11. **Semiotic Safety** - Cultural appropriateness
12. **Narrative Physics** - Story logic
13. **Integrity & Grounding** - Factual accuracy
14. **Visual Orchestration** - Visual elements
15. **Logic-Emotion Balance** - Persuasion balance
16. **Adaptive Persona Fusion** - Final alignment

### **Control Tower Priority**

When rules conflict, this order applies:
1. Safety (cultural, emotional, symbolic)
2. User Intent
3. Persona Protection
4. Platform Compatibility
5. Narrative Fidelity
6. Structural Logic
7. Emotional Voltage
8. CTA Accuracy

---

## 🛠️ Troubleshooting

### **Problem: "Engine KB not available"**

**Solution:**
```bash
python engine_kb_builder.py
```

### **Problem: "MIMIR rules loaded: 0 chars"**

**Cause:** Engine KB exists but query returned no results

**Solution:**
1. Check if PDFs are in `Engine rules/01-Content-Generation/`
2. Rebuild Engine KB: `python engine_kb_builder.py`
3. Test queries: `python engine_kb_builder.py test`

### **Problem: Content quality hasn't improved**

**Check:**
1. Verify MIMIR rules are being loaded (check console output)
2. Ensure Engine KB was built successfully
3. Check if PDFs contain actual content (not scanned images)

### **Problem: Generation is slower**

**Expected:** MIMIR queries add 1-2 seconds per generation

**If too slow:**
- Check Ollama is running (for embeddings)
- Reduce `k` parameter in queries (default is 3-5)
- Consider caching frequently used rules

---

## 📈 Measuring MIMIR Impact

### **Quality Metrics to Track**

**Before MIMIR:**
- Generic language
- Platform mismatches
- Weak CTAs
- Inconsistent tone

**After MIMIR:**
- Persona-aligned language
- Platform-optimized format
- Strong, contextual CTAs
- Consistent brand voice

### **A/B Testing**

Generate content with and without MIMIR:

```python
# Without MIMIR (disable Engine KB)
generator.engine_kb = None
content_without = generator.generate_linkedin_post(...)

# With MIMIR (enable Engine KB)
generator.engine_kb = EngineKBHelper()
content_with = generator.generate_linkedin_post(...)

# Compare engagement metrics
```

---

## 🔄 Updating MIMIR Rules

### **Adding New Rules**

1. Add PDF to `Engine rules/01-Content-Generation/Parts/`
2. Update `INDEX.md` with new part details
3. Rebuild Engine KB:
   ```bash
   python engine_kb_builder.py
   ```

### **Modifying Existing Rules**

1. Edit the PDF in `Engine rules/01-Content-Generation/`
2. Rebuild Engine KB:
   ```bash
   python engine_kb_builder.py
   ```

---

## 🎯 Best Practices

### **1. Always Build Engine KB First**
```bash
python engine_kb_builder.py
```

### **2. Test After Building**
```bash
python test_mimir_integration.py
```

### **3. Monitor Console Output**
Look for MIMIR rule loading messages

### **4. Use Appropriate Tone Settings**
- Professional: CXO, enterprise content
- Bold: Disruptive, attention-grabbing
- Casual: Approachable, conversational

### **5. Match Platform to Content Type**
- Long-form analysis → LinkedIn Article
- Quick insights → LinkedIn Post
- Punchy statements → Twitter
- Deep dives → Blog
- Visual storytelling → YouTube

---

## 📚 Additional Resources

- **MIMIR System Overview:** `Engine rules/01-Content-Generation/INDEX.md`
- **Integration Summary:** `MIMIR_INTEGRATION_SUMMARY.md`
- **Test Script:** `test_mimir_integration.py`
- **Engine KB Helper:** `engine_kb_helper.py`
- **Engine KB Builder:** `engine_kb_builder.py`

---

## 🎉 Success Indicators

You'll know MIMIR is working when you see:

✅ More professional, polished content  
✅ Better persona alignment  
✅ Platform-optimized formatting  
✅ Reduced clichés and weak language  
✅ Consistent brand voice  
✅ Improved engagement metrics  
✅ Factually accurate content  
✅ Culturally appropriate messaging  

---

**Last Updated:** November 14, 2025  
**Version:** 1.0  
**Status:** Production Ready
