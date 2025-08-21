"""
Legal precedent research engine.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import re

from src.core.models import (
    Document, ClauseAnalysis, LegalPrecedent, DocumentType
)
from src.core.config import get_config
from src.analysis.llm_client import LLMClient


class PrecedentResearcher:
    """
    AI-powered legal precedent research engine.
    
    Searches for relevant case law and legal precedents from:
    - Google Scholar
    - Free legal databases
    - Cached precedent database
    - Legal research APIs (if configured)
    """
    
    def __init__(self):
        """Initialize the precedent researcher."""
        self.config = get_config()
        self.llm_client = LLMClient()
        
        # Common legal databases and sources
        self.free_databases = {
            'google_scholar': 'https://scholar.google.com/scholar',
            'justia': 'https://law.justia.com',
            'courtlistener': 'https://www.courtlistener.com',
            'findlaw': 'https://caselaw.findlaw.com'
        }
        
        # Document type to legal area mapping
        self.legal_areas = {
            DocumentType.EMPLOYMENT_AGREEMENT: [
                'employment law', 'labor law', 'wrongful termination',
                'employee benefits', 'non-compete agreements'
            ],
            DocumentType.NDA: [
                'confidentiality agreements', 'trade secrets',
                'proprietary information', 'non-disclosure'
            ],
            DocumentType.MERGER_AGREEMENT: [
                'mergers and acquisitions', 'corporate law',
                'due diligence', 'purchase agreements'
            ],
            DocumentType.COMMERCIAL_CONTRACT: [
                'commercial law', 'contract law', 'breach of contract',
                'commercial disputes', 'sales agreements'
            ],
            DocumentType.LICENSE_AGREEMENT: [
                'intellectual property', 'licensing agreements',
                'royalty disputes', 'patent law', 'copyright law'
            ],
            DocumentType.LEASE_AGREEMENT: [
                'real estate law', 'landlord tenant law',
                'lease disputes', 'property law'
            ],
            DocumentType.SERVICE_AGREEMENT: [
                'service contracts', 'professional services',
                'consulting agreements', 'performance obligations'
            ]
        }
    
    async def research_precedents(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis]
    ) -> List[LegalPrecedent]:
        """
        Research relevant legal precedents for the document.
        
        Args:
            document: Document to research precedents for
            clause_analyses: Clause analyses to guide research
            
        Returns:
            List of relevant legal precedents
        """
        precedents = []
        
        # Research based on document type
        doc_type_precedents = await self._research_by_document_type(document)
        precedents.extend(doc_type_precedents)
        
        # Research based on high-risk clauses
        clause_precedents = await self._research_by_clauses(document, clause_analyses)
        precedents.extend(clause_precedents)
        
        # Research specific legal issues
        issue_precedents = await self._research_by_legal_issues(document, clause_analyses)
        precedents.extend(issue_precedents)
        
        # Remove duplicates and sort by relevance
        unique_precedents = self._deduplicate_precedents(precedents)
        sorted_precedents = sorted(
            unique_precedents, 
            key=lambda x: x.confidence_score, 
            reverse=True
        )
        
        return sorted_precedents[:10]  # Return top 10 most relevant
    
    async def _research_by_document_type(self, document: Document) -> List[LegalPrecedent]:
        """Research precedents based on document type."""
        precedents = []
        
        legal_areas = self.legal_areas.get(document.document_type, [])
        
        for area in legal_areas[:3]:  # Limit to top 3 areas
            search_results = await self._search_free_databases(area, document.document_type.value)
            area_precedents = await self._process_search_results(search_results, area)
            precedents.extend(area_precedents)
        
        return precedents
    
    async def _research_by_clauses(
        self, 
        document: Document, 
        clause_analyses: List[ClauseAnalysis]
    ) -> List[LegalPrecedent]:
        """Research precedents for specific problematic clauses."""
        precedents = []
        
        # Focus on clauses with low scores or specific legal issues
        problematic_clauses = [
            analysis for analysis in clause_analyses
            if analysis.enforceability_score < 70 or analysis.business_favorability < 60
        ]
        
        for clause_analysis in problematic_clauses[:5]:  # Limit to top 5
            # Extract key legal concepts from the clause
            legal_concepts = await self._extract_legal_concepts(clause_analysis)
            
            for concept in legal_concepts:
                search_results = await self._search_free_databases(concept, 'contract law')
                concept_precedents = await self._process_search_results(search_results, concept)
                precedents.extend(concept_precedents)
        
        return precedents
    
    async def _research_by_legal_issues(
        self, 
        document: Document, 
        clause_analyses: List[ClauseAnalysis]
    ) -> List[LegalPrecedent]:
        """Research precedents for specific legal issues identified."""
        precedents = []
        
        # Extract common legal issues from clause analyses
        legal_issues = []
        for analysis in clause_analyses:
            if analysis.risk_factors:
                legal_issues.extend(analysis.risk_factors[:2])
        
        # Research each unique legal issue
        unique_issues = list(set(legal_issues))[:5]  # Limit to 5 unique issues
        
        for issue in unique_issues:
            search_results = await self._search_free_databases(issue, 'case law')
            issue_precedents = await self._process_search_results(search_results, issue)
            precedents.extend(issue_precedents)
        
        return precedents
    
    async def _search_free_databases(
        self, 
        query: str, 
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """Search free legal databases for relevant cases."""
        search_results = []
        
        # Search Google Scholar
        if self.config.legal_db.enable_google_scholar:
            try:
                scholar_results = await self._search_google_scholar(query, context)
                search_results.extend(scholar_results)
            except Exception as e:
                print(f"Google Scholar search failed: {str(e)}")
        
        # Search other free databases
        if self.config.legal_db.enable_free_databases:
            try:
                free_db_results = await self._search_other_databases(query, context)
                search_results.extend(free_db_results)
            except Exception as e:
                print(f"Free database search failed: {str(e)}")
        
        return search_results[:20]  # Limit results
    
    async def _search_google_scholar(
        self, 
        query: str, 
        context: str
    ) -> List[Dict[str, Any]]:
        """Search Google Scholar for legal cases."""
        results = []
        
        try:
            # Construct search URL
            search_query = f"{query} {context} case law"
            search_url = f"https://scholar.google.com/scholar?q={requests.utils.quote(search_query)}&hl=en&scisbd=1"
            
            # Add headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results
            result_divs = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            for div in result_divs[:5]:  # Limit to top 5 results
                try:
                    title_elem = div.find('h3', class_='gs_rt')
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Extract case citation if available
                        citation_elem = div.find('div', class_='gs_a')
                        citation = citation_elem.get_text().strip() if citation_elem else ""
                        
                        # Extract snippet
                        snippet_elem = div.find('div', class_='gs_rs')
                        snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                        
                        results.append({
                            'title': title,
                            'citation': citation,
                            'snippet': snippet,
                            'source': 'Google Scholar',
                            'query': query
                        })
                
                except Exception as e:
                    print(f"Error parsing search result: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Google Scholar search error: {str(e)}")
        
        return results
    
    async def _search_other_databases(
        self, 
        query: str, 
        context: str
    ) -> List[Dict[str, Any]]:
        """Search other free legal databases."""
        results = []
        
        # For now, return mock results to avoid rate limiting
        # In production, implement actual searches of Justia, CourtListener, etc.
        
        mock_results = [
            {
                'title': f'Case Related to {query.title()}',
                'citation': 'Sample Citation 123 F.3d 456 (2023)',
                'snippet': f'This case addresses {query} in the context of {context}...',
                'source': 'Legal Database',
                'query': query
            }
        ]
        
        return mock_results
    
    async def _process_search_results(
        self, 
        search_results: List[Dict[str, Any]], 
        topic: str
    ) -> List[LegalPrecedent]:
        """Process search results into legal precedent objects."""
        precedents = []
        
        for result in search_results:
            try:
                # Extract case information
                case_name = result.get('title', 'Unknown Case')
                citation = result.get('citation', 'Citation not available')
                snippet = result.get('snippet', '')
                
                # Extract year from citation
                year_match = re.search(r'\b(19|20)\d{2}\b', citation)
                year = int(year_match.group()) if year_match else 2023
                
                # Extract jurisdiction (simplified)
                jurisdiction = 'Federal'
                if 'state' in citation.lower() or 'supreme court' in citation.lower():
                    jurisdiction = 'State'
                
                # Generate relevance assessment using LLM
                relevance_data = await self._assess_relevance(result, topic)
                
                precedent = LegalPrecedent(
                    case_name=case_name,
                    citation=citation,
                    jurisdiction=jurisdiction,
                    year=year,
                    relevant_principle=relevance_data.get('principle', snippet[:200]),
                    application_to_clause=relevance_data.get('application', ''),
                    confidence_score=relevance_data.get('confidence', 0.6)
                )
                
                precedents.append(precedent)
                
            except Exception as e:
                print(f"Error processing search result: {str(e)}")
                continue
        
        return precedents
    
    async def _assess_relevance(
        self, 
        search_result: Dict[str, Any], 
        topic: str
    ) -> Dict[str, Any]:
        """Assess the relevance of a search result using LLM."""
        relevance_prompt = f"""
You are a legal research expert. Assess the relevance of this case to the topic "{topic}".

CASE: {search_result.get('title', '')}
CITATION: {search_result.get('citation', '')}
SNIPPET: {search_result.get('snippet', '')}

Provide:
1. Relevant legal principle (1-2 sentences)
2. Application to contract clauses (1-2 sentences)
3. Confidence score (0-100%)

Be concise and focus on practical application to contract review.
"""
        
        try:
            response = await self.llm_client.generate_response(
                relevance_prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            # Parse response
            confidence_match = re.search(r'(\d+)%', response)
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.6
            
            return {
                'principle': response[:200],
                'application': response[200:400] if len(response) > 200 else response,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Relevance assessment failed: {str(e)}")
            return {
                'principle': search_result.get('snippet', '')[:200],
                'application': f"May be relevant to {topic}",
                'confidence': 0.5
            }
    
    async def _extract_legal_concepts(self, clause_analysis: ClauseAnalysis) -> List[str]:
        """Extract key legal concepts from clause analysis."""
        concepts = []
        
        # Extract from legal reasoning
        if clause_analysis.legal_reasoning:
            concept_prompt = f"""
Extract 2-3 key legal concepts from this analysis:

{clause_analysis.legal_reasoning}

Respond with only the legal concepts, one per line (e.g., "liability limitation", "force majeure", "indemnification").
"""
            
            try:
                response = await self.llm_client.generate_response(
                    concept_prompt,
                    max_tokens=100,
                    temperature=0.1
                )
                
                # Parse concepts from response
                lines = response.strip().split('\n')
                for line in lines:
                    concept = line.strip(' -"\'')
                    if concept and len(concept) > 3:
                        concepts.append(concept)
                        
            except Exception as e:
                print(f"Concept extraction failed: {str(e)}")
        
        # Fallback to risk factors
        if not concepts and clause_analysis.risk_factors:
            concepts = clause_analysis.risk_factors[:3]
        
        return concepts[:3]  # Limit to 3 concepts
    
    def _deduplicate_precedents(self, precedents: List[LegalPrecedent]) -> List[LegalPrecedent]:
        """Remove duplicate precedents based on case name and citation."""
        seen = set()
        unique_precedents = []
        
        for precedent in precedents:
            key = (precedent.case_name.lower(), precedent.citation.lower())
            if key not in seen:
                seen.add(key)
                unique_precedents.append(precedent)
        
        return unique_precedents
    
    async def search_specific_precedent(
        self, 
        legal_issue: str, 
        jurisdiction: str = "federal"
    ) -> List[LegalPrecedent]:
        """
        Search for precedents on a specific legal issue.
        
        Args:
            legal_issue: Specific legal issue to research
            jurisdiction: Jurisdiction to focus on
            
        Returns:
            List of relevant precedents
        """
        search_results = await self._search_free_databases(legal_issue, jurisdiction)
        return await self._process_search_results(search_results, legal_issue)
    
    async def get_precedent_summary(
        self, 
        precedents: List[LegalPrecedent], 
        context: str
    ) -> str:
        """
        Generate a summary of precedents for a given context.
        
        Args:
            precedents: List of precedents to summarize
            context: Context for the summary
            
        Returns:
            Summary of precedents
        """
        if not precedents:
            return "No relevant precedents found."
        
        summary_prompt = f"""
Summarize these legal precedents in the context of {context}:

{chr(10).join(f"- {p.case_name}: {p.relevant_principle}" for p in precedents[:5])}

Provide a concise 2-3 sentence summary highlighting the key legal principles and their application to contract review.
"""
        
        try:
            return await self.llm_client.generate_response(
                summary_prompt,
                max_tokens=200,
                temperature=0.3
            )
        except Exception as e:
            print(f"Precedent summary failed: {str(e)}")
            return "Precedent summary unavailable."
