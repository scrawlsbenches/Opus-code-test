# Customer Service Sample Corpus

## Overview

This directory contains realistic customer service documents designed to demonstrate the Cortical Text Processor's capabilities in a non-technical domain. The corpus includes FAQs, troubleshooting guides, policy documents, and response templates that showcase natural language understanding, semantic relationships, and query expansion features.

## Contents

### New Markdown Documents (8 files)

**FAQ Documents:**
- **faq-billing.md** - Billing, payments, invoices, subscriptions, and financial questions
- **faq-shipping.md** - Shipping methods, delivery, tracking, and international orders

**Troubleshooting Guides:**
- **troubleshoot-login.md** - Account access, password issues, two-factor authentication
- **troubleshoot-payment.md** - Payment failures, declined cards, promotional codes

**Policy Documents:**
- **policy-returns.md** - Return eligibility, refund process, exchange procedures
- **policy-privacy.md** - Data collection, privacy rights, security measures

**Response Templates:**
- **template-apology.md** - Service failure apologies and customer recovery
- **template-resolution.md** - Issue resolution confirmations and follow-ups

### Legacy Text Documents (14 files)

**FAQ Documents (4 files):**
- software_product_faq.txt - CloudSync Pro cloud storage
- electronics_product_faq.txt - TechGear wireless earbuds
- subscription_service_faq.txt - StreamFlix entertainment service
- account_management_faq.txt - Account access and management

**Troubleshooting Guides (4 files):**
- software_installation_troubleshooting.txt
- connectivity_network_troubleshooting.txt
- payment_billing_troubleshooting.txt
- login_access_troubleshooting.txt

**Policy Documents (3 files):**
- return_refund_policy.txt
- shipping_delivery_policy.txt
- privacy_data_security_policy.txt

**Templates and Guidelines (3 files):**
- email_response_templates.txt
- service_level_agreement.txt
- customer_feedback_survey_guide.txt

## Purpose

This sample cluster demonstrates the Cortical Text Processor's ability to:

1. **Cluster Related Concepts** - Group similar support topics (billing issues, shipping problems, account access)
2. **Expand Queries** - Find synonyms and related terms (refund/reimbursement, cancel/terminate, issue/problem)
3. **Extract Semantic Relations** - Understand relationships (refund relates to return relates to policy)
4. **Retrieve Relevant Information** - Find answers to customer questions across multiple documents
5. **Understand Intent** - Parse natural language queries into actionable search terms

## Indexing and Searching

### Quick Start

```bash
# Index the customer service corpus
python -c "
from cortical.processor import CorticalTextProcessor
import os

processor = CorticalTextProcessor()

# Load all customer service documents
cs_dir = 'samples/customer_service'
for filename in os.listdir(cs_dir):
    if filename.endswith(('.md', '.txt')) and filename != 'README.md':
        filepath = os.path.join(cs_dir, filename)
        with open(filepath, 'r') as f:
            text = f.read()
        doc_id = f'cs_{filename.rsplit(\".\", 1)[0]}'  # Remove extension
        processor.process_document(doc_id, text)

# Compute all relationships and rankings
processor.compute_all()

# Save the indexed corpus
processor.save('customer_service_corpus.pkl')
print(f'Indexed {processor.document_count} customer service documents')
"
```

### Interactive Search

```python
from cortical.processor import CorticalTextProcessor

# Load the indexed corpus
processor = CorticalTextProcessor.load('customer_service_corpus.pkl')

# Search for customer questions
results = processor.find_documents_for_query("how do I get a refund")
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")

# See query expansion
expanded = processor.expand_query("cancel my order")
print("Expanded terms:", expanded)

# Find relevant passages
passages = processor.find_passages_for_query("my package is late", top_n=3)
for passage, score, doc_id in passages:
    print(f"\nFrom {doc_id} (score: {score:.3f}):")
    print(passage[:200] + "...")
```

## Example Queries

### Billing and Payment Questions

```python
# These queries should return relevant billing documents
queries = [
    "how do I change my payment method",
    "I was charged twice",
    "cancel my subscription",
    "download my invoice",
    "update billing address",
    "dispute a charge",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = processor.find_documents_for_query(query, top_n=3)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.3f}")
```

**Expected Results**: Should retrieve `faq-billing`, `policy-returns`, and `troubleshoot-payment` documents.

### Shipping and Delivery Questions

```python
queries = [
    "where is my package",
    "track my order",
    "shipping costs too high",
    "international delivery",
    "package arrived damaged",
    "change delivery address",
]
```

**Expected Results**: Should primarily retrieve `faq-shipping` and `shipping_delivery_policy` documents.

### Account Access Questions

```python
queries = [
    "forgot my password",
    "can't log in",
    "account locked",
    "enable two factor authentication",
    "reset my password",
    "SSO not working",
]
```

**Expected Results**: Should retrieve `troubleshoot-login` and `login_access_troubleshooting` documents.

### Return and Refund Questions

```python
queries = [
    "how to return an item",
    "get my money back",
    "exchange for different size",
    "return window expired",
    "where's my refund",
    "return shipping cost",
]
```

**Expected Results**: Should retrieve `policy-returns` and `return_refund_policy` documents.

## Demonstrating Key Features

### 1. Query Expansion

The processor should identify synonyms and related terms:

```python
# See how the processor expands customer service terms
terms_to_expand = [
    "refund",      # Should expand to: reimbursement, money back, credit, return
    "cancel",      # Should expand to: terminate, discontinue, end, stop
    "problem",     # Should expand to: issue, error, trouble, difficulty
    "late",        # Should expand to: delayed, overdue, slow, behind
    "broken",      # Should expand to: defective, damaged, faulty, not working
]

for term in terms_to_expand:
    expanded = processor.expand_query(term, max_expansions=5)
    print(f"\n{term} →")
    for exp_term, weight in sorted(expanded.items(), key=lambda x: -x[1])[:5]:
        print(f"  {exp_term}: {weight:.3f}")
```

### 2. Concept Clustering

View clusters of related support topics:

```python
# Build concept clusters
processor.build_concept_clusters(resolution=1.0)

# View clusters
from cortical.layers import CorticalLayer
concepts = processor.layers[CorticalLayer.CONCEPTS]

print(f"\nFound {concepts.column_count()} concept clusters:")
for concept_id, minicolumn in list(concepts.minicolumns.items())[:10]:
    terms = concept_id.replace('L2_', '').split('_')[:5]
    print(f"  {', '.join(terms)} ({len(minicolumn.document_ids)} docs)")
```

**Expected Clusters**:
- Billing/payment/subscription/invoice
- Shipping/delivery/tracking/package
- Return/refund/exchange/policy
- Login/password/authentication/access
- Support/help/contact/service

### 3. Semantic Relations

Extract relationships between customer service concepts:

```python
# Extract semantic relations
processor.extract_corpus_semantics()

# View key relations
print("\nTop semantic relations:")
for term1, relation, term2, weight in processor.semantic_relations[:20]:
    print(f"  {term1} --{relation}--> {term2} ({weight:.2f})")
```

**Expected Relations**:
- refund → relates_to → return
- payment → requires → billing
- tracking → enables → delivery
- password → enables → login
- policy → governs → return

### 4. Intent Understanding

Parse natural language customer questions:

```python
# Test intent parsing
customer_questions = [
    "where do we explain our return policy",
    "how do customers reset their password",
    "what should I do if payment fails",
    "can I change my shipping address",
    "why was my account locked",
]

for question in customer_questions:
    intent = processor.parse_intent_query(question)
    print(f"\nQuestion: {question}")
    print(f"  Intent: {intent.get('intent', 'unknown')}")
    print(f"  Subject: {intent.get('subject', 'N/A')}")
    print(f"  Action: {intent.get('action', 'N/A')}")
```

### 5. Passage Retrieval (RAG)

Find specific answer passages for customer questions:

```python
# Retrieve relevant passages for RAG systems
questions = [
    "What payment methods do you accept?",
    "How long do refunds take?",
    "Can I return an item after 30 days?",
    "What if my package shows delivered but I didn't get it?",
]

for question in questions:
    print(f"\nQ: {question}")
    passages = processor.find_passages_for_query(question, top_n=2, chunk_size=200)
    for passage, score, doc_id in passages:
        print(f"\nA (from {doc_id}, score: {score:.3f}):")
        print(passage)
```

## Domain-Specific Vocabulary

This corpus demonstrates handling of customer service terminology:

### Operational Terms
- **SLA** (Service Level Agreement)
- **Escalation** (routing to higher support tier)
- **Ticket** (support case identifier)
- **Resolution** (issue fix/answer)
- **Hold** (temporary account status)
- **Verification** (identity confirmation)

### Process Terms
- **Fulfillment** (order processing and shipping)
- **Restocking fee** (charge for returns)
- **Chargeback** (payment dispute)
- **Provisioning** (account setup)
- **Reconciliation** (payment matching)

### Customer Journey Terms
- **Onboarding** (initial setup)
- **Retention** (keeping customers)
- **Churn** (customer departure)
- **Conversion** (completing purchase)
- **Touchpoint** (interaction point)

## Cross-Document Semantic Queries

Test queries that should retrieve information from multiple related documents:

- **"Security and privacy protection"** → privacy_data_security_policy, policy-privacy, troubleshoot-login (2FA)
- **"Delivery problems and solutions"** → faq-shipping, shipping_delivery_policy, template-resolution
- **"Account access issues"** → troubleshoot-login, login_access_troubleshooting, account_management_faq
- **"Payment and billing concerns"** → faq-billing, payment_billing_troubleshooting, policy-returns

## Integration Examples

### Chatbot Integration

```python
class CustomerServiceChatbot:
    def __init__(self, corpus_path):
        self.processor = CorticalTextProcessor.load(corpus_path)

    def answer_question(self, question):
        """Find relevant passages to answer customer question."""
        passages = self.processor.find_passages_for_query(
            question,
            top_n=3,
            chunk_size=300
        )

        if not passages:
            return "I couldn't find information about that. Please contact support."

        # Return most relevant passage
        best_passage, score, doc_id = passages[0]

        if score < 0.3:
            return "I'm not sure about that. Let me connect you with a specialist."

        return best_passage

    def suggest_related_articles(self, query):
        """Suggest helpful articles based on query."""
        docs = self.processor.find_documents_for_query(query, top_n=5)
        return [doc_id.replace('cs_', '').replace('-', ' ').title()
                for doc_id, score in docs if score > 0.2]

# Usage
bot = CustomerServiceChatbot('customer_service_corpus.pkl')
answer = bot.answer_question("How do I get a refund?")
print(answer)

related = bot.suggest_related_articles("shipping problems")
print("Related articles:", related)
```

### Support Ticket Classification

```python
def classify_ticket(ticket_text, processor):
    """Classify support ticket into category."""
    # Find most relevant documents
    results = processor.find_documents_for_query(ticket_text, top_n=3)

    if not results:
        return "general"

    top_doc, score = results[0]

    # Map document to category
    categories = {
        'billing': ['faq-billing', 'payment'],
        'shipping': ['faq-shipping', 'delivery'],
        'returns': ['policy-returns', 'refund'],
        'account': ['troubleshoot-login', 'login'],
        'privacy': ['policy-privacy', 'privacy'],
    }

    for category, patterns in categories.items():
        if any(pattern in top_doc for pattern in patterns):
            return category

    return "general"

# Test classification
tickets = [
    "My payment was declined and I don't know why",
    "Package hasn't arrived and it's been 2 weeks",
    "Need to return an item but lost the receipt",
    "Can't log into my account, forgot password",
]

for ticket in tickets:
    category = classify_ticket(ticket, processor)
    print(f"{category.upper()}: {ticket}")
```

## Performance Benchmarks

Expected performance on typical hardware (reference):

- **Indexing**: ~2-4 seconds for all 22 documents
- **Query Search**: <50ms for simple queries
- **Passage Retrieval**: <100ms with chunking
- **Query Expansion**: <10ms
- **Concept Clustering**: ~2-5 seconds (one-time)

## Use Cases

This corpus is suitable for demonstrating:

- **Customer support automation** - Chatbots, virtual assistants
- **Knowledge base search** - Help centers, FAQ search
- **Ticket routing** - Automatic categorization and assignment
- **Answer suggestion** - Support agent assistance tools
- **Content recommendation** - Related article suggestions
- **Self-service portals** - Customer account management
- **Training data** - For customer service ML models

## Contributing

To add more customer service documents:

1. **Follow naming convention**: `category-topic.md` or `descriptive_name.txt`
2. **Use realistic language**: Authentic customer service tone and terminology
3. **Include variety**: Different question types, solutions, and scenarios
4. **Cross-reference**: Link related topics naturally in content
5. **Test searchability**: Verify new docs are retrieved for relevant queries

## Statistics

- **Total documents**: 22 (8 markdown + 14 text)
- **Document types**: FAQs, troubleshooting, policies, templates
- **Coverage**: Billing, shipping, returns, privacy, authentication, technical support
- **Use cases**: Search, classification, chatbots, RAG, recommendations

---

**Questions or suggestions?** This corpus is designed to showcase natural language processing capabilities beyond code search. Feedback on search quality, relevance ranking, and additional use cases is welcome.
