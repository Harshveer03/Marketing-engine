import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()

VECTOR_DB_DIR = "./vectordb"
OUTPUT_FILE = "./generated/niche_icp.json"

# ----------- PROMPT -------------
PROMPT_TEMPLATE = """You are an AI assistant that extracts structured business insights from documents about a company's niche and Ideal Customer Profile (ICP).

Hard rules:
- PRESERVE key industry terminology, tool names, and specific business concepts exactly as they appear
- Rephrase only for clarity, but maintain critical keywords and technical terms
- Include specific methodologies, technologies, and market terms mentioned in the document
- Produce fact-based, granular, and actionable reasons for problems ("why") and concrete mechanisms for solutions ("how")
- If a fact cannot be inferred, leave it as "" for strings or [] for lists
- Output ONLY valid JSON matching the schema below. No extra text, no commentary

Keyword Preservation Priority:
- Keep exact technology names (e.g., "Gong", "Clari", "HubSpot", "Salesforce", "Outreach", "ZoomInfo")
- Preserve industry acronyms (e.g., "GTM", "RevOps", "CRM", "SaaS", "B2B", "PLG", "ABM")
- Maintain specific methodologies (e.g., "pipeline forecasting", "conversion optimization", "lead scoring")
- Include market terminology (e.g., "predictable revenue", "scalable growth", "product-led growth")
- Preserve measurement terms and KPI language exactly as written

Industry Context Requirements:
- Extract and preserve all mentions of specific business processes, tools, and methodologies
- Include competitive landscape terms and market positioning language
- Maintain technical terminology that would appear in industry publications
- Preserve measurement terms and KPI language exactly as written

From the provided context, extract and preserve the following fields with industry-specific terminology.

Schema & required content expectations:
- industry: (string) concise category using exact industry terminology, e.g., "B2B SaaS Go-To-Market", "Enterprise Sales Automation", "RevOps Technology"
- target_audience: (list of strings) Include specific role titles, company types, and industry segments exactly as mentioned. Combine roles with industry context (e.g., "B2B SaaS Founders", "GTM Leaders in lean sales teams", "RevOps Directors at growth-stage companies")
- customer_pain_points: (list of objects) each object MUST include:
  - challenge: (string) Include specific business terms, tool names, and measurable outcomes from the document. Use industry terminology that would appear in news headlines or trend articles
  - why: (list of objects) each object is a specific possible cause with:
    - cause: (string) concise label using industry terminology (e.g., "poor CRM data instrumentation", "GTM stack fragmentation")
    - explanation: (string) 1-3 sentences explaining the causal mechanism using specific tools, processes, and industry terms
    - indicators: (list of strings) concrete signals using industry KPIs and metrics (e.g., "Salesforce conversion drops at demo stage", "Gong call analysis shows objection patterns")
    - recommended_first_checks: (list of short actions) immediate checks using specific tools and processes (e.g., "audit last 30 HubSpot records for missing UTM sources")
- customer_needs: (list of objects) each object MUST include:
  - need: (string) Use specific, searchable terminology that includes technology names, business processes, and measurable outcomes mentioned in the document
  - how: (list of objects) each object is a concrete way the company can meet that need:
    - approach: (string) short name using industry terminology (e.g., "RevOps instrumentation + Mixpanel analytics layer")
    - details: (string) 1-3 sentences describing implementation using specific tools, integration points, and industry processes
    - measurable_signs: (list of strings) KPIs using industry-standard metrics (e.g., "pipeline velocity +15%", "Gong conversation intelligence score >8.5")
    - first_deliverables: (list of strings) immediate outputs using industry terminology (e.g., "Salesforce dashboard with 5 leading GTM indicators", "RevOps playbook with 20 prioritized conversion optimizations")
- value_proposition: (string) 1 sentence using industry terminology and specific business outcomes mentioned in the document
- brand_tone: (string) inferred communication style using industry context, e.g., "Strategic GTM Thought-leadership", "Data-driven RevOps Authority"

Output format (must match exactly):
{{
  "industry": "",
  "target_audience": [],
  "customer_pain_points": [
    {{
      "challenge": "",
      "why": [
        {{
          "cause": "",
          "explanation": "",
          "indicators": [],
          "recommended_first_checks": []
        }}
      ]
    }}
  ],
  "customer_needs": [
    {{
      "need": "",
      "how": [
        {{
          "approach": "",
          "details": "",
          "measurable_signs": [],
          "first_deliverables": []
        }}
      ]
    }}
  ],
  "value_proposition": "",
  "brand_tone": ""
}}

Context:
{context}"""




# ----------- FUNCTIONS -------------
def load_faiss_index():
    """Load existing FAISS index"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectordb

def query_icp(vectordb, query="Summarize the niche and ICP details"):
    """Query FAISS index for ICP context"""
    docs = vectordb.similarity_search(query, k=10)
    return "\n\n".join([doc.page_content for doc in docs])

def extract_structured_info(context: str, model="models/gemini-2.5-flash") -> dict:
    """Use LLM to extract structured JSON from context"""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    prompt = PromptTemplate(input_variables=["context"], template=PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=JsonOutputParser())
    response = chain.run({"context": context})
    
    # Parse into dict if string
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except:
            raise ValueError("⚠️ LLM did not return valid JSON.")
    return response

def save_json(data: dict, output_file: str = OUTPUT_FILE):
    """Save extracted JSON"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved structured info to {output_file}")

# ----------- MAIN -------------
if __name__ == "__main__":
    print("🚀 Generating niche_icp.json from embeddings...")
    vectordb = load_faiss_index()
    context = query_icp(vectordb)
    structured_data = extract_structured_info(context)
    save_json(structured_data)
    print("🎉 niche_icp.json created successfully.")
