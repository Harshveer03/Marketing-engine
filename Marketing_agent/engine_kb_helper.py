"""
Engine Knowledge Base Helper
Query MIMIR rules during content generation
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class EngineKBHelper:
    """
    Helper to fetch relevant MIMIR rules from Engine KB
    """
    
    def __init__(self, engine_kb_path="./engine_kb/vectordb"):
        self.engine_kb_path = engine_kb_path
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Load Engine KB if it exists
        if os.path.exists(engine_kb_path):
            self.vectordb = FAISS.load_local(
                engine_kb_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Engine KB loaded successfully")
        else:
            self.vectordb = None
            print("⚠️ Engine KB not found. Run engine_kb_builder.py first.")
    
    def query(self, query_text, k=3):
        """
        Generic query method to fetch relevant MIMIR rules
        
        Args:
            query_text: What to search for
            k: Number of results to return
        
        Returns:
            str: Combined relevant rules
        """
        if not self.vectordb:
            return "Engine KB not available"
        
        print(f"\n🔍 Engine KB Query:")
        print(f"   Search: '{query_text[:100]}...'")
        print(f"   Fetching top {k} results...")
        
        docs = self.vectordb.similarity_search(query_text, k=k)
        
        print(f"\n📚 Retrieved {len(docs)} MIMIR rule chunks:")
        
        # Combine results with source attribution
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            part_type = doc.metadata.get('type', 'unknown')
            content = doc.page_content.strip()
            
            print(f"   {i}. {source} ({part_type})")
            print(f"      Preview: {content[:100]}...")
            
            results.append(f"[{source}]\n{content}")
        
        combined = "\n\n---\n\n".join(results)
        print(f"\n✅ Total MIMIR rules: {len(combined)} characters\n")
        
        return combined
    
    # Specific helper methods for different content types
    
    def get_blog_rules(self, topic, audience, tone):
        """Get MIMIR rules for blog generation"""
        query = f"""
        Blog content generation rules for:
        - Topic: {topic}
        - Audience: {audience}
        - Tone: {tone}
        
        Include: intent, tone guidelines, persona tailoring, narrative structure, grounding rules
        """
        return self.query(query, k=5)
    
    def get_social_rules(self, platform, topic, audience, tone):
        """Get MIMIR rules for social media generation"""
        query = f"""
        Social media content rules for {platform}:
        - Topic: {topic}
        - Audience: {audience}
        - Tone: {tone}
        
        Include: structural tailoring, platform fit, tone adaptation, persona matching
        """
        return self.query(query, k=4)
    
    def get_tone_guidelines(self, tone, audience):
        """Get specific tone guidelines"""
        query = f"Tone guidelines for {tone} tone targeting {audience} audience"
        return self.query(query, k=2)
    
    def get_persona_rules(self, persona_type, industry):
        """Get persona tailoring rules"""
        query = f"Persona tailoring principles for {persona_type} in {industry} industry"
        return self.query(query, k=3)
    
    def get_visual_rules(self, content_type):
        """Get visual orchestration rules"""
        query = f"Visual orchestration rules for {content_type} - faithful vs cinematic modes"
        return self.query(query, k=2)
    
    def get_grounding_rules(self):
        """Get integrity and grounding rules"""
        query = "Integrity and grounding law - prevent hallucination and drift"
        return self.query(query, k=2)
    
    def get_anti_patterns(self):
        """Get anti-patterns to avoid"""
        query = "Anti-patterns removal - weak phrasing, clichés, mistakes to avoid"
        return self.query(query, k=2)
    
    def get_narrative_rules(self, content_type):
        """Get narrative physics rules"""
        query = f"Narrative physics for {content_type} - story logic, rhythm, pacing, flow"
        return self.query(query, k=2)
    
    def get_logic_emotion_balance(self, content_purpose):
        """Get logic-emotion balance rules"""
        query = f"Logic-emotion balance for {content_purpose} - voltage calibration, consequence stacking"
        return self.query(query, k=2)
    
    def get_quality_criteria(self, content_type):
        """Get quality scoring criteria"""
        query = f"Quality criteria and validation rules for {content_type}"
        return self.query(query, k=3)
    
    def get_control_tower_sequence(self):
        """Get the Control Tower execution sequence"""
        query = "Control Tower execution sequence - 16 part order and validation"
        return self.query(query, k=1)
    
    def get_linkedin_content_structure(self, content_type, tone, persona, industry, topic, challenge=None):
        """
        Query LinkedIn Content Guide for specific structure and template
        
        Args:
            content_type: "post" or "article"
            tone: "sharp", "reflective", "teaching", "contrarian", "neutral", etc.
            persona: "SDR", "AE", "CRO", "Founder", "CXO", etc.
            industry: "SaaS", "B2B SaaS", "Travel", "Retail", "Fintech", etc.
            topic: The main topic/subject of the content
            challenge: Optional specific challenge/problem
        
        Returns:
            str: LinkedIn Content Guide structure with post types, skeletons, templates, and sections
        """
        if not self.vectordb:
            return ""
        
        # Build comprehensive query for fuzzy matching
        query_parts = [
            f"LinkedIn Content Guide for {content_type}:",
            f"- Topic: {topic}",
            f"- Tone: {tone}" if tone else "",
            f"- Target Persona: {persona}" if persona else "",
            f"- Industry: {industry}" if industry else "",
            f"- Challenge: {challenge}" if challenge else "",
            "",
            "Include:",
            "- Post type selection (narrative, jolt, insight, contrarian, teaching)",
            "- Skeleton selection from 50 skeletons",
            "- Template with section prompts (hook, context, insight, story, consequence, shift, close)",
            "- Section-specific guidance for each part",
            "- Voice and formatting rules"
        ]
        
        query = "\n".join([p for p in query_parts if p])
        
        print(f"\n🔍 LinkedIn Content Guide Query:")
        print(f"   Content Type: {content_type}")
        print(f"   Tone: {tone or 'default'}")
        print(f"   Persona: {persona or 'general'}")
        print(f"   Industry: {industry or 'universal'}")
        print(f"   Topic: {topic[:50]}...")
        print(f"   Fetching structure...")
        
        # Query with higher k for better fuzzy matching
        docs = self.vectordb.similarity_search(query, k=8)
        
        print(f"\n📚 Retrieved {len(docs)} LinkedIn Guide chunks")
        
        # Combine results with source attribution
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content.strip()
            
            print(f"   {i}. {source}")
            print(f"      Preview: {content[:100]}...")
            
            results.append(content)
        
        combined = "\n\n---\n\n".join(results)
        print(f"\n✅ LinkedIn Content Guide structure: {len(combined)} characters\n")
        
        return combined


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Engine KB Helper\n")
    
    helper = EngineKBHelper()
    
    if helper.vectordb:
        # Test 1: Blog rules
        print("="*60)
        print("Test 1: Blog Generation Rules")
        print("="*60)
        rules = helper.get_blog_rules(
            topic="AI in Healthcare",
            audience="Healthcare CXOs",
            tone="professional"
        )
        print(rules[:500] + "...\n")
        
        # Test 2: Tone guidelines
        print("="*60)
        print("Test 2: Tone Guidelines")
        print("="*60)
        rules = helper.get_tone_guidelines("professional", "B2B executives")
        print(rules[:500] + "...\n")
        
        # Test 3: Visual rules
        print("="*60)
        print("Test 3: Visual Orchestration Rules")
        print("="*60)
        rules = helper.get_visual_rules("blog featured image")
        print(rules[:500] + "...\n")
        
        # Test 4: Grounding rules
        print("="*60)
        print("Test 4: Grounding Rules")
        print("="*60)
        rules = helper.get_grounding_rules()
        print(rules[:500] + "...\n")
        
        print("✅ All tests complete!")
    else:
        print("❌ Engine KB not available. Run engine_kb_builder.py first.")
