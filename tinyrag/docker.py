#!/usr/bin/env python3
"""
Build and run Docker container with appropriate environment config.
Extracts AWS credentials only when bedrock services are used.
"""

import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv


def get_aws_credentials_from_profile(profile_name: str):
    """Extract AWS credentials from a profile, including STS identity and expiry."""
    credentials = {}

    try:
        session = boto3.Session(profile_name=profile_name)
        creds = session.get_credentials()

        if not creds:
            raise ValueError(f"No credentials found for profile '{profile_name}'")

        if not creds.access_key or not creds.secret_key:
            raise ValueError(f"Incomplete credentials for profile '{profile_name}'")

        credentials["AWS_ACCESS_KEY_ID"] = creds.access_key
        credentials["AWS_SECRET_ACCESS_KEY"] = creds.secret_key

        if creds.token:
            credentials["AWS_SESSION_TOKEN"] = creds.token

        region = session.region_name
        if region:
            credentials["AWS_DEFAULT_REGION"] = region
            credentials["AWS_REGION"] = region

        sts = session.client("sts")
        identity = sts.get_caller_identity()
        if identity:
            credentials["AWS_ACCOUNT_ID"] = identity.get("Account", "")
            credentials["AWS_USER_ARN"] = identity.get("Arn", "")

        if hasattr(creds, "token"):
            frozen_creds = creds.get_frozen_credentials()
            if hasattr(frozen_creds, "expiry_time") and frozen_creds.expiry_time:
                credentials["AWS_CREDENTIALS_EXPIRY"] = (
                    frozen_creds.expiry_time.isoformat()
                )

    except Exception as e:
        raise ValueError(f"Error getting AWS credentials: {str(e)}") from None

    return credentials


def write_env_file(file_path: Path, config_vars: dict):
    with open(file_path, "w") as f:
        for key, value in sorted(config_vars.items()):
            if value:
                f.write(f"{key}={value}\n")


def main():
    image_name = "tinyrag"

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    env_file = project_root / ".env"
    env_docker_file = project_root / ".env.docker"

    if not env_file.exists():
        print(f"Error: .env file not found at {env_file}", file=sys.stderr)
        sys.exit(1)

    load_dotenv(env_file)

    chat_service = os.getenv("CHAT_SERVICE", "openai")
    embed_service = os.getenv("EMBED_SERVICE", "openai")

    env_vars = {
        "CHAT_SERVICE": chat_service,
        "EMBED_SERVICE": embed_service,
    }

    uses_bedrock = chat_service == "bedrock" or embed_service == "bedrock"
    uses_openai = chat_service == "openai" or embed_service == "openai"
    uses_groq = chat_service == "groq" or embed_service == "groq"

    try:
        if uses_bedrock:
            profile_name = os.getenv("AWS_PROFILE")
            if profile_name:
                print(f"Extracting credentials from AWS profile: {profile_name}")
                aws_creds = get_aws_credentials_from_profile(profile_name)
                env_vars.update(aws_creds)
            else:
                for key, value in os.environ.items():
                    if key.startswith("AWS_"):
                        env_vars[key] = value

        if uses_openai:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                env_vars["OPENAI_API_KEY"] = openai_api_key

        if uses_groq:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                env_vars["GROQ_API_KEY"] = groq_api_key

        write_env_file(env_docker_file, env_vars)
        print(f"Successfully wrote config to {env_docker_file}")
        print(f"Extracted {len(env_vars)} environment variables")

        build_cmd = ["docker", "build", "-t", image_name, "."]
        print("\nBuilding Docker image:")
        print(" ".join(build_cmd))
        sys.stdout.flush()
        subprocess.run(build_cmd, check=True, cwd=str(project_root))

        docker_cmd = [
            "docker",
            "run",
            "-p",
            "80:80",
            "--env-file",
            str(env_docker_file),
            image_name,
        ]

        print("\nRunning Docker command:")
        print(" ".join(docker_cmd))
        print()

        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
