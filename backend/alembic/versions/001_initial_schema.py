"""Initial schema: tables, pgvector, match_documents.

Revision ID: 001
Revises:
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("create extension if not exists vector")

    # Document chunks for RAG
    op.execute("""
        create table if not exists document_chunks (
            id uuid default gen_random_uuid() primary key,
            content text not null,
            metadata jsonb default '{}',
            embedding vector(1536),
            source_doc_id text,
            chunk_index integer,
            created_at timestamptz default now()
        )
    """)

    # Index for similarity search (bump maintenance_work_mem for IVFFlat build)
    op.execute("SET maintenance_work_mem = '64MB'")
    op.execute("""
        create index if not exists ix_document_chunks_embedding
        on document_chunks
        using ivfflat (embedding vector_cosine_ops)
        with (lists = 10)
    """)

    # Prompt versions
    op.execute("""
        create table if not exists prompt_versions (
            id uuid default gen_random_uuid() primary key,
            name text not null,
            version integer not null,
            template text not null,
            is_active boolean default false,
            weight float default 0.0,
            metadata jsonb default '{}',
            created_at timestamptz default now(),
            unique(name, version)
        )
    """)

    # Eval results
    op.execute("""
        create table if not exists eval_results (
            id uuid default gen_random_uuid() primary key,
            run_id text not null,
            dataset_name text not null,
            scores jsonb not null,
            details jsonb default '{}',
            created_at timestamptz default now()
        )
    """)

    # Match function for similarity search
    op.execute("""
        create or replace function match_documents(
            query_embedding vector(1536),
            match_threshold float,
            match_count int
        )
        returns table (
            id uuid,
            content text,
            metadata jsonb,
            similarity float
        )
        language sql stable
        as $$
            select
                document_chunks.id,
                document_chunks.content,
                document_chunks.metadata,
                1 - (document_chunks.embedding <=> query_embedding) as similarity
            from document_chunks
            where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
            order by (document_chunks.embedding <=> query_embedding)
            limit match_count;
        $$;
    """)


def downgrade() -> None:
    op.execute("drop function if exists match_documents(vector, float, int)")
    op.execute("drop table if exists eval_results")
    op.execute("drop table if exists prompt_versions")
    op.execute("drop table if exists document_chunks")
    op.execute("drop extension if exists vector")
