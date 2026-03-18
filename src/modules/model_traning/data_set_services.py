import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.modules.data.services.article_repository import ArticleRepository
from src.core.database import get_async_db_session
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, load_dataset


class TrainingDatasetBuilder:
    """Build training datasets from database using ArticleRepository."""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.article_repo = ArticleRepository()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_all_paragraphs(self, output_file: str = "training_data.jsonl"):
        """
        Fetch all paragraphs and save to JSONL file.
        
        Format:
        {
            "id": article_id,
            "title": article_title,
            "url": article_url,
            "prompt": "merged paragraphs content",
            "completion": "meta_description or title"
        }
        """
        async with get_async_db_session() as session:
            articles = await self.article_repo.get_all_articles_with_paragraphs(session)
            
            output_path = self.output_dir / output_file
            saved_count = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in tqdm(articles, desc="Processing articles", unit="article"):
                    if article.seo_meta_keywords == "" or article.seo_meta_keywords is None:
                        continue
                    
                    # Merge all paragraphs into prompt
                    paragraphs_text = "\n\n".join([
                        p.content 
                        for p in sorted(article.paragraphs, key=lambda x: x.order_index)
                    ])
                    
                    # Use meta keywords as completion
                    completion = article.seo_meta_keywords
                    
                    record = {
                        "id": article.id,
                        "title": article.title,
                        "url": article.url,
                        "prompt": paragraphs_text,
                        "completion": completion
                    }
                    
                    # Write as JSONL (one JSON object per line)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved_count += 1
            
            print(f"\n✓ Saved {saved_count} articles (out of {len(articles)} total) to {output_path}")
            return output_path
    
    def load_and_inspect_dataset(self, file_path: str, num_samples: int = 5):
        """
        Load and inspect a dataset file.
        
        Args:
            file_path: Path to the dataset file (JSONL format)
            num_samples: Number of sample records to display
        
        Returns:
            Dataset object for further processing
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        print(f"Loading dataset from {file_path}...")
        dataset = load_dataset('json', data_files=str(file_path), split='train')
        
        print(f"\n{'='*60}")
        print(f"Dataset Overview")
        print(f"{'='*60}")
        print(f"Total records: {len(dataset)}")
        print(f"\nFeatures:")
        for feature, dtype in dataset.features.items():
            print(f"  - {feature}: {dtype}")
        
        # Statistics
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        
        if 'prompt' in dataset.features:
            prompt_lengths = [len(record['prompt']) for record in dataset]
            print(f"\nPrompt character lengths:")
            print(f"  - Min: {min(prompt_lengths)}")
            print(f"  - Max: {max(prompt_lengths)}")
            print(f"  - Average: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
        
        if 'completion' in dataset.features:
            completion_lengths = [len(record['completion']) for record in dataset]
            print(f"\nCompletion character lengths:")
            print(f"  - Min: {min(completion_lengths)}")
            print(f"  - Max: {max(completion_lengths)}")
            print(f"  - Average: {sum(completion_lengths) / len(completion_lengths):.2f}")
        
        # Sample records
        print(f"\n{'='*60}")
        print(f"Sample Records (showing {min(num_samples, len(dataset))} of {len(dataset)})")
        print(f"{'='*60}")
        
        for i, record in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
            print(f"\n--- Record {i+1} ---")
            for key, value in record.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}... (truncated, total {len(value)} chars)")
                else:
                    print(f"{key}: {value}")
        
        print(f"\n{'='*60}\n")
        return dataset
    
    def load_and_inspect_remote_dataset(self, repo_id: str, token: Optional[str] = None, num_samples: int = 5, split: str = 'train'):
        """
        Load and inspect a dataset from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID in format 'username/repo-name'
            token: Hugging Face API token (if not provided, uses HF_TOKEN env var)
            num_samples: Number of sample records to display
            split: Dataset split to inspect ('train', 'validation', 'test')
        
        Returns:
            Dataset object for further processing
        """
        # Get token from parameter or environment
        hf_token = token or os.getenv("HF_TOKEN")
        
        print(f"Loading dataset from Hugging Face Hub: {repo_id}...")
        try:
            # Load dataset from HF Hub
            if hf_token:
                dataset = load_dataset(repo_id, split=split, token=hf_token)
            else:
                dataset = load_dataset(repo_id, split=split)
            
            print(f"\n{'='*60}")
            print(f"Remote Dataset Overview")
            print(f"{'='*60}")
            print(f"Repository: {repo_id}")
            print(f"Split: {split}")
            print(f"Total records: {len(dataset)}")
            print(f"\nFeatures:")
            for feature, dtype in dataset.features.items():
                print(f"  - {feature}: {dtype}")
            
            # Statistics
            print(f"\n{'='*60}")
            print(f"Dataset Statistics")
            print(f"{'='*60}")
            
            if 'prompt' in dataset.features:
                prompt_lengths = [len(record['prompt']) for record in dataset]
                print(f"\nPrompt character lengths:")
                print(f"  - Min: {min(prompt_lengths)}")
                print(f"  - Max: {max(prompt_lengths)}")
                print(f"  - Average: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
            
            if 'completion' in dataset.features:
                completion_lengths = [len(record['completion']) for record in dataset]
                print(f"\nCompletion character lengths:")
                print(f"  - Min: {min(completion_lengths)}")
                print(f"  - Max: {max(completion_lengths)}")
                print(f"  - Average: {sum(completion_lengths) / len(completion_lengths):.2f}")
            
            # Sample records
            print(f"\n{'='*60}")
            print(f"Sample Records (showing {min(num_samples, len(dataset))} of {len(dataset)})")
            print(f"{'='*60}")
            
            for i, record in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                print(f"\n--- Record {i+1} ---")
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"{key}: {value[:200]}... (truncated, total {len(value)} chars)")
                    else:
                        print(f"{key}: {value}")
            
            print(f"\n{'='*60}\n")
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset from {repo_id}: {e}")
            raise
    
    def upload_to_huggingface(
        self,
        file_path: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
        commit_message: str = "Upload training dataset"
    ):
        """
        Upload dataset file to Hugging Face Hub as a proper dataset.
        
        Args:
            file_path: Path to the dataset file to upload (JSONL format)
            repo_id: Repository ID in format 'username/repo-name'
            token: Hugging Face API token (if not provided, uses HF_TOKEN env var)
            private: Whether to create a private repository
            commit_message: Commit message for the upload
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Get token from parameter or environment
        hf_token = token or os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Hugging Face token required. Provide via --token argument or HF_TOKEN environment variable"
            )
        
        print(f"Creating/accessing repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=hf_token,
                exist_ok=True
            )
            print(f"✓ Repository ready: {repo_id}")
        except Exception as e:
            print(f"Repository creation skipped or failed: {e}")
        
        print(f"Loading dataset from {file_path.name}...")
        # Load the JSONL file as a dataset
        dataset = load_dataset('json', data_files=str(file_path), split='train')
        
        print(f"Dataset info:")
        print(f"  - Total examples: {len(dataset)}")
        print(f"  - Features: {dataset.features}")
        
        # Split into train (80%) and validation (20%)
        print(f"\nSplitting dataset into train and validation (80/20 split)...")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        print(f"  - Training examples: {len(dataset['train'])}")
        print(f"  - Validation examples: {len(dataset['test'])}")
        
        # Rename 'test' split to 'validation'
        dataset_dict = {
            'train': dataset['train'],
            'validation': dataset['test']
        }
        
        print(f"\nPushing dataset to {repo_id}...")
        from datasets import DatasetDict
        DatasetDict(dataset_dict).push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            commit_message=commit_message,
            private=private
        )
        
        print(f"\n✓ Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
        print(f"  - Train split: {len(dataset_dict['train'])} examples")
        print(f"  - Validation split: {len(dataset_dict['validation'])} examples")


async def generate_from_db(output_dir: str = "./datasets", output_file: str = "training_data.jsonl"):
    """Generate training dataset from database."""
    builder = TrainingDatasetBuilder(output_dir=output_dir)
    output_path = await builder.fetch_all_paragraphs(output_file)
    
    print(f"Dataset saved to: {output_path}")


def inspect_dataset(file_path: str, num_samples: int = 5):
    """Load and inspect a dataset file."""
    builder = TrainingDatasetBuilder()
    builder.load_and_inspect_dataset(file_path=file_path, num_samples=num_samples)


def inspect_remote_dataset(repo_id: str, token: Optional[str] = None, num_samples: int = 5, split: str = 'train'):
    """Load and inspect a dataset from Hugging Face Hub."""
    builder = TrainingDatasetBuilder()
    builder.load_and_inspect_remote_dataset(repo_id=repo_id, token=token, num_samples=num_samples, split=split)


def upload_to_hf(
    file_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload training dataset"
):
    """Upload dataset to Hugging Face."""
    builder = TrainingDatasetBuilder()
    builder.upload_to_huggingface(
        file_path=file_path,
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message
    )


async def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Training Dataset Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # generate-from-db command
    parser_generate = subparsers.add_parser(
        'generate-from-db',
        help='Generate training dataset from database articles'
    )
    parser_generate.add_argument(
        '--output-dir',
        type=str,
        default='./datasets',
        help='Output directory for dataset files (default: ./datasets)'
    )
    parser_generate.add_argument(
        '--output-file',
        type=str,
        default='training_data.jsonl',
        help='Output filename (default: training_data.jsonl)'
    )
    
    # inspect-dataset command
    parser_inspect = subparsers.add_parser(
        'inspect-dataset',
        help='Load and inspect a dataset file'
    )
    parser_inspect.add_argument(
        'file_path',
        type=str,
        help='Path to the dataset file to inspect'
    )
    parser_inspect.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of sample records to display (default: 5)'
    )
    
    # inspect-remote command
    parser_inspect_remote = subparsers.add_parser(
        'inspect-remote',
        help='Load and inspect a dataset from Hugging Face Hub'
    )
    parser_inspect_remote.add_argument(
        'repo_id',
        type=str,
        help='Hugging Face repository ID (format: username/repo-name)'
    )
    parser_inspect_remote.add_argument(
        '--token',
        type=str,
        help='Hugging Face API token (or set HF_TOKEN env variable)'
    )
    parser_inspect_remote.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of sample records to display (default: 5)'
    )
    parser_inspect_remote.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Dataset split to inspect (default: train)'
    )
    
    # upload-to-hf command
    parser_upload = subparsers.add_parser(
        'upload-to-hf',
        help='Upload dataset file to Hugging Face Hub'
    )
    parser_upload.add_argument(
        'file_path',
        type=str,
        help='Path to the dataset file to upload'
    )
    parser_upload.add_argument(
        'repo_id',
        type=str,
        help='Hugging Face repository ID (format: username/repo-name)'
    )
    parser_upload.add_argument(
        '--token',
        type=str,
        help='Hugging Face API token (or set HF_TOKEN env variable)'
    )
    parser_upload.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    parser_upload.add_argument(
        '--commit-message',
        type=str,
        default='Upload training dataset',
        help='Commit message for the upload'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate-from-db':
        await generate_from_db(args.output_dir, args.output_file)
    elif args.command == 'inspect-dataset':
        inspect_dataset(file_path=args.file_path, num_samples=args.num_samples)
    elif args.command == 'inspect-remote':
        inspect_remote_dataset(
            repo_id=args.repo_id,
            token=args.token,
            num_samples=args.num_samples,
            split=args.split
        )
    elif args.command == 'upload-to-hf':
        upload_to_hf(
            file_path=args.file_path,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())


#python -msrc/modules/model_traning/data_set_services.py generate-from-db --output-dir ./my-datasets --output-file my_data.jsonl

#python -m src/modules/model_traning/data_set_services.py upload-to-hf ./datasets/training_data.jsonl username/my-dataset

#python -m src.modules.model_traning.data_set_services inspect-remote ibrahim-nada/xxxxxx