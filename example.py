"""
Example usage of the Legal Document Review AI Agent.

This script demonstrates how to use the agent programmatically
to analyze legal documents.
"""

import asyncio
import tempfile
import os
from pathlib import Path

from src.agent import LegalDocumentReviewAgent
from src.core.models import DocumentType


async def main():
    """Main example function."""
    
    print("🏛️ Legal Document Review AI Agent - Example Usage")
    print("=" * 60)
    
    # Initialize the agent
    print("\n1. Initializing Legal AI Agent...")
    try:
        agent = LegalDocumentReviewAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("💡 Make sure you have API keys configured in .env file")
        return
    
    # Create a sample employment agreement
    print("\n2. Creating sample employment agreement...")
    sample_contract = create_sample_employment_agreement()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_contract)
        temp_path = f.name
    
    print(f"📄 Sample document created: {temp_path}")
    
    try:
        # Perform full document review
        print("\n3. Starting comprehensive document review...")
        print("   This may take 30-60 seconds depending on API response times...")
        
        result = await agent.review_document(
            document_path=temp_path,
            document_type=DocumentType.EMPLOYMENT_AGREEMENT,
            custom_instructions="Focus on employee protection and fair terms",
            priority_areas=["financial", "termination", "intellectual_property"]
        )
        
        # Display results
        print("\n🎉 Analysis completed!")
        print("=" * 60)
        
        # Executive Summary
        print(f"\n📋 EXECUTIVE SUMMARY")
        print(f"Document Type: {result.executive_summary.document_type.value.replace('_', ' ').title()}")
        print(f"Overall Assessment: {result.executive_summary.overall_assessment}")
        print(f"Recommendation: {result.executive_summary.recommendation}")
        print(f"Business Impact: {result.executive_summary.business_impact}")
        print(f"Estimated Review Time: {result.executive_summary.estimated_review_time}")
        
        # Risk Assessment
        print(f"\n⚠️ RISK ASSESSMENT")
        print(f"Overall Risk Score: {result.risk_assessment.overall_score}/100")
        print(f"Critical Issues: {len(result.risk_assessment.critical_issues)}")
        print(f"Total Issues: {len(result.risk_assessment.risk_issues)}")
        
        if result.risk_assessment.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in result.risk_assessment.critical_issues[:3]:
                print(f"  • {issue.title}: {issue.description[:100]}...")
        
        # Key Recommendations
        if result.risk_assessment.recommendations:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for rec in result.risk_assessment.recommendations:
                print(f"  • {rec}")
        
        # Clause Analysis Summary
        print(f"\n📝 CLAUSE ANALYSIS")
        print(f"Clauses Analyzed: {len(result.clause_analyses)}")
        
        if result.clause_analyses:
            low_enforceability = [
                c for c in result.clause_analyses 
                if c.enforceability_score < 70
            ]
            low_favorability = [
                c for c in result.clause_analyses 
                if c.business_favorability < 60
            ]
            
            print(f"Low Enforceability Clauses: {len(low_enforceability)}")
            print(f"Unfavorable Business Terms: {len(low_favorability)}")
        
        # Redline Summary
        print(f"\n✏️ REDLINED CHANGES")
        print(f"Total Changes Suggested: {len(result.redlined_document.changes)}")
        
        change_types = {}
        for change in result.redlined_document.changes:
            change_type = change.change_type.value
            if change_type not in change_types:
                change_types[change_type] = 0
            change_types[change_type] += 1
        
        for change_type, count in change_types.items():
            print(f"  • {change_type.replace('_', ' ').title()}: {count}")
        
        # Legal Precedents
        print(f"\n📚 LEGAL PRECEDENTS")
        print(f"Relevant Precedents Found: {len(result.legal_precedents)}")
        
        if result.legal_precedents:
            print(f"\nTop Precedents:")
            for precedent in result.legal_precedents[:3]:
                print(f"  • {precedent.case_name} ({precedent.year})")
                print(f"    Principle: {precedent.relevant_principle[:100]}...")
        
        # Processing Statistics
        print(f"\n📊 PROCESSING STATISTICS")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        print(f"Confidence Score: {int(result.confidence_score * 100)}%")
        print(f"Workflow Steps: {len(result.workflow_steps)}")
        
        print(f"\n✅ Example completed successfully!")
        
        # Quick risk assessment example
        print(f"\n" + "=" * 60)
        print("🚀 QUICK RISK ASSESSMENT EXAMPLE")
        print("=" * 60)
        
        quick_result = await agent.quick_risk_assessment(temp_path)
        
        print(f"Overall Risk Score: {quick_result.overall_score}/100")
        print(f"Total Issues: {len(quick_result.risk_issues)}")
        print(f"Critical Issues: {len(quick_result.critical_issues)}")
        
        if quick_result.critical_issues:
            print(f"\nQuick Critical Issues:")
            for issue in quick_result.critical_issues[:2]:
                print(f"  • {issue.title}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("💡 This might be due to missing API keys or network issues")
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        print(f"\n🧹 Cleaned up temporary files")


def create_sample_employment_agreement() -> str:
    """Create a sample employment agreement with various risk levels."""
    
    return """
EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on [DATE] between TechCorp Inc., a Delaware corporation ("Company"), and [EMPLOYEE NAME] ("Employee").

1. POSITION AND DUTIES
Employee shall serve as Senior Software Engineer and shall perform such duties as assigned by the Company. Employee agrees to devote full-time efforts to the Company and shall not engage in any other employment or business activities without prior written consent.

2. COMPENSATION AND BENEFITS
2.1 Base Salary: Employee shall receive an annual base salary of $120,000, payable in accordance with Company's standard payroll practices.
2.2 Bonus: Employee may be eligible for an annual performance bonus at the sole discretion of the Company.
2.3 Benefits: Employee shall be entitled to participate in Company benefit plans as generally made available to similarly situated employees.

3. INTELLECTUAL PROPERTY
Employee agrees that all inventions, discoveries, and improvements made during employment shall be the exclusive property of the Company. Employee hereby assigns all rights, title, and interest in such intellectual property to the Company.

4. CONFIDENTIALITY
Employee acknowledges access to confidential and proprietary information and agrees to maintain strict confidentiality during and after employment. This obligation shall survive termination of this Agreement indefinitely.

5. NON-COMPETE AND NON-SOLICITATION
Employee agrees that for a period of two (2) years following termination, Employee shall not:
(a) Engage in any business competitive with the Company within a 100-mile radius;
(b) Solicit or attempt to solicit any customers or employees of the Company;
(c) Work for any competitor of the Company.

6. TERMINATION
6.1 At-Will Employment: Either party may terminate this Agreement at any time, with or without cause and with or without notice.
6.2 Company Termination: If Company terminates Employee without cause, Company shall provide two weeks' severance pay.
6.3 Immediate Termination: Company may terminate Employee immediately for cause, including but not limited to misconduct, breach of this Agreement, or poor performance.

7. LIABILITY AND INDEMNIFICATION
Employee agrees to indemnify and hold harmless the Company from any and all claims, damages, losses, and expenses arising out of Employee's performance of duties or breach of this Agreement. Employee's liability under this provision shall be unlimited.

8. DISPUTE RESOLUTION
Any disputes arising under this Agreement shall be resolved exclusively through binding arbitration in Delaware. Employee waives any right to jury trial and agrees that the Company shall not be liable for attorney's fees regardless of the outcome.

9. GOVERNING LAW
This Agreement shall be governed by the laws of Delaware without regard to conflict of law principles.

10. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties and may only be modified in writing signed by both parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

TECHCORP INC.

By: _________________________
Name: [CEO NAME]
Title: Chief Executive Officer

EMPLOYEE:

_________________________
[EMPLOYEE NAME]
"""


if __name__ == "__main__":
    print("🚀 Starting Legal Document Review AI Agent Example...")
    
    # Run the example
    asyncio.run(main())
    
    print("\n" + "=" * 60)
    print("📖 NEXT STEPS:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run the FastAPI server: python -m src.api.main")
    print("3. Run the Streamlit UI: streamlit run src/ui/app.py")
    print("4. Upload real legal documents for analysis")
    print("\n🎯 Happy legal document reviewing!")
