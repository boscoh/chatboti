#!/usr/bin/env python3
"""
Build and run Docker container with appropriate environment config.
Extracts AWS credentials only when bedrock services are used.
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def log(msg: str):
    """Log info message."""
    print(f"[INFO] {msg}")


def warn(msg: str):
    """Log warning message."""
    print(f"[WARN] {msg}", file=sys.stderr)


def error(msg: str):
    """Log error message and exit."""
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def get_aws_config(is_raise_exception: bool = True):
    """Get AWS configuration for boto3 client initialization.

    Falls back to boto3's credential chain if AWS_PROFILE doesn't exist.

    :param is_raise_exception: Raise exceptions or warn on errors
    :return: Dict with profile_name and region_name keys
    """
    load_dotenv()

    aws_config = {}
    available_profiles = set()
    credentials_path = os.path.expanduser("~/.aws/credentials")
    config_path = os.path.expanduser("~/.aws/config")

    if os.path.exists(credentials_path):
        import configparser
        config = configparser.ConfigParser()
        config.read(credentials_path)
        available_profiles.update(config.sections())

    if os.path.exists(config_path):
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        for section in config.sections():
            if section.startswith("profile "):
                available_profiles.add(section[8:])
            elif section != "default":
                available_profiles.add(section)

    profile_name = os.getenv("AWS_PROFILE")
    profile_not_found = False
    if profile_name:
        if profile_name in available_profiles:
            aws_config["profile_name"] = profile_name
        else:
            log(f"AWS profile '{profile_name}' not found, using default credential chain...")
            profile_not_found = True

    region = os.getenv("AWS_REGION")
    if region:
        aws_config["region_name"] = region

    if profile_not_found:
        os.environ.pop("AWS_PROFILE", None)

    try:
        session = boto3.Session(**aws_config)
        credentials = session.get_credentials()

        if not credentials:
            if is_raise_exception:
                if available_profiles:
                    error(
                        f"No AWS credentials found.\n"
                        f"Available profiles: {', '.join(available_profiles)}\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
                else:
                    error(
                        f"No AWS credentials found.\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
            return aws_config

        sts = session.client("sts")
        sts.get_caller_identity()

        if profile_name and profile_name in aws_config.get("profile_name", ""):
            config_path = os.path.expanduser("~/.aws/config")
            if os.path.exists(config_path):
                import configparser
                config = configparser.ConfigParser()
                config.read(config_path)
                section = f"profile {profile_name}"
                if config.has_section(section) and config.has_option(section, "sso_start_url"):
                    if hasattr(credentials, "token"):
                        creds = credentials.get_frozen_credentials()
                        if hasattr(creds, "expiry_time") and creds.expiry_time < datetime.now(timezone.utc):
                            login_cmd = f"aws sso login --profile {profile_name}"
                            error(f"AWS SSO session expired. Please run:\n  {login_cmd}")

        return aws_config
    except ClientError as e:
        if is_raise_exception:
            raise
        error_code = e.response["Error"]["Code"]

        if error_code == "ExpiredToken":
            # Check if SSO to provide better error message
            profile_to_check = aws_config.get("profile_name", profile_name)
            if profile_to_check:
                config_path = os.path.expanduser("~/.aws/config")
                if os.path.exists(config_path):
                    import configparser
                    config = configparser.ConfigParser()
                    config.read(config_path)
                    section = f"profile {profile_to_check}"
                    if config.has_section(section) and config.has_option(section, "sso_start_url"):
                        login_cmd = f"aws sso login --profile {profile_to_check}"
                        warn(f"AWS SSO session expired. Please run:\n  {login_cmd}")
                        return aws_config
            warn("AWS credentials have expired")
        elif error_code == "InvalidClientTokenId":
            warn("AWS credentials are invalid. Please reconfigure:\n  aws configure")
        else:
            warn(f"AWS API error: {error_code}")
    except Exception as e:
        if is_raise_exception:
            raise
        warn(f"AWS credential check failed: {e}")

    return aws_config


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
    image_name = "chatboti"

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    env_file = project_root / ".env"
    env_docker_file = project_root / ".env.docker"

    if not env_file.exists():
        error(f".env file not found at {env_file}")

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
            # Validate AWS configuration first
            aws_config = get_aws_config(is_raise_exception=True)

            # Extract credentials if we have a profile or fallback to env vars
            if aws_config.get("profile_name"):
                profile_name = aws_config["profile_name"]
                log(f"Extracting credentials from AWS profile: {profile_name}")
                aws_creds = get_aws_credentials_from_profile(profile_name)
                env_vars.update(aws_creds)
            else:
                # Fall back to environment variable credentials
                log("Using AWS credentials from environment variables")
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
        log(f"Successfully wrote config to {env_docker_file}")
        log(f"Extracted {len(env_vars)} environment variables")

        build_cmd = ["docker", "build", "-t", image_name, "."]
        log("\nBuilding Docker image:")
        log(" ".join(build_cmd))
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

        log("\nRunning Docker command:")
        log(" ".join(docker_cmd))
        print()

        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        error(f"Command failed with exit code {e.returncode}: {e.cmd}")
    except Exception as e:
        error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
