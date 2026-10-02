import numpy as np

from legal_agent.agent import NOT_FOUND, UNVERIFIED_ANSWER, LegalAgent
from legal_agent.documents import SourceRef, TextChunk
from legal_agent.retrieval import ContractIndex


class FakeClient:
    def __init__(self, query_embedding, answer="The contract states the term. [Evidence 1]"):
        self.query_embedding = query_embedding
        self.answers = answer if isinstance(answer, list) else [answer]
        self.generated = False
        self.user_prompt = ""
        self.chat_calls = 0

    def embed(self, texts):
        return [self.query_embedding for _ in texts]

    def chat(self, _system_prompt, user_prompt):
        self.generated = True
        self.user_prompt = user_prompt
        answer = self.answers[min(self.chat_calls, len(self.answers) - 1)]
        self.chat_calls += 1
        return answer


def test_agent_abstains_without_relevant_evidence():
    chunk = TextChunk("The agreement renews each year.", (SourceRef("a.pdf", "PDF page 1"),))
    index = ContractIndex((chunk,), np.array([[1.0, 0.0]], dtype=np.float32))
    client = FakeClient([-1.0, 0.0])

    answer = LegalAgent(client, index, min_score=0.28).ask("What is the liability cap?")

    assert answer.text == NOT_FOUND
    assert answer.evidence == ()
    assert client.generated is False


def test_agent_returns_retrieved_evidence_with_answer():
    source = SourceRef("a.pdf", "PDF page 2")
    chunk = TextChunk("The agreement renews each year.", (source,))
    index = ContractIndex((chunk,), np.array([[1.0, 0.0]], dtype=np.float32))
    client = FakeClient([1.0, 0.0])

    answer = LegalAgent(client, index).ask("When does the agreement renew?")

    assert "Evidence 1" in answer.text
    assert answer.evidence[0].chunk.sources == (source,)


def test_agent_withholds_uncited_or_invalidly_cited_answers():
    chunk = TextChunk("The agreement renews each year.", (SourceRef("a.pdf", "PDF page 1"),))
    index = ContractIndex((chunk,), np.array([[1.0, 0.0]], dtype=np.float32))

    for model_answer in ("It renews annually.", "It renews annually. [Evidence 8]"):
        client = FakeClient([1.0, 0.0], answer=model_answer)
        answer = LegalAgent(client, index).ask("When does it renew?")

        assert answer.text == UNVERIFIED_ANSWER
        assert answer.evidence


def test_party_question_includes_opening_and_cites_uncited_answer():
    cover = TextChunk("Master Services Agreement", (SourceRef("a.pdf", "PDF page 1"),))
    parties = TextChunk(
        "This Agreement is between Redwood Example LLC (Customer) and Harborlight "
        "Example Inc. (Provider).",
        (SourceRef("a.pdf", "PDF page 1"),),
    )
    detail = TextChunk("Invoices are due within thirty days.", (SourceRef("a.pdf", "PDF page 3"),))
    index = ContractIndex(
        (cover, parties, detail), np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    )
    client = FakeClient(
        [1.0, 0.0],
        answer=(
            "Redwood Example LLC is the Customer, and Harborlight Example Inc. "
            "is the Provider."
        ),
    )

    answer = LegalAgent(client, index).ask("What parties are involved?")

    assert answer.evidence[0].chunk == parties
    assert answer.text.endswith("[Evidence 1]")
    assert "full legal name" in client.user_prompt


def test_agent_retries_once_to_repair_missing_citation():
    chunk = TextChunk("The agreement renews each year.", (SourceRef("a.pdf", "PDF page 1"),))
    index = ContractIndex((chunk,), np.array([[1.0, 0.0]], dtype=np.float32))
    client = FakeClient(
        [1.0, 0.0],
        answer=[
            "The agreement renews for one-year terms.",
            "The agreement renews for one-year terms. [Evidence 1]",
        ],
    )

    answer = LegalAgent(client, index).ask("How does renewal work?")

    assert answer.text.endswith("[Evidence 1]")
    assert client.chat_calls == 2
