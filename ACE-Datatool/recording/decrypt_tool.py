import argparse
import os
import struct
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
except Exception as error:  # pylint: disable=broad-except
    Cipher = None
    CRYPTO_ERROR = str(error)
else:
    CRYPTO_ERROR = ""

DEFAULT_ENCRYPTION_KEY = b"my_secret_key_32_bytes_long_abc!"
CHUNK_SIZE = 4 * 1024 * 1024
SALT_SIZE = 16


def _derive_file_key(master_key: bytes, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"ace-file-encryption",
    ).derive(master_key)


def _parse_key(raw_value: Optional[str]) -> Optional[bytes]:
    if not raw_value:
        return DEFAULT_ENCRYPTION_KEY

    cleaned = raw_value.strip()
    key: Optional[bytes] = None
    if len(cleaned) == 64:
        try:
            key = bytes.fromhex(cleaned)
        except ValueError:
            key = None

    if key is None:
        key = cleaned.encode("utf-8")

    if len(key) != 32:
        print(f"Invalid key length: {len(key)} (expected 32 bytes)")
        return None

    return key


def decrypt_single_file(encrypted_file_path: str, output_dir: str, key: bytes) -> None:
    try:
        with open(encrypted_file_path, "rb") as source:
            salt = source.read(SALT_SIZE)
            if len(salt) < SALT_SIZE:
                return

            file_key = _derive_file_key(key, salt)

            nonce = source.read(12)
            if len(nonce) < 12:
                return

            metadata_len_bytes = source.read(2)
            if not metadata_len_bytes:
                return

            metadata_len = struct.unpack("!H", metadata_len_bytes)[0]
            encrypted_metadata = source.read(metadata_len)

            file_size = os.path.getsize(encrypted_file_path)
            header_size = SALT_SIZE + 12 + 2 + metadata_len
            ciphertext_len = file_size - header_size - 16
            if ciphertext_len < 0:
                print(
                    "[SKIP] {} invalid header size".format(
                        os.path.basename(encrypted_file_path)
                    )
                )
                return

            tag_offset = header_size + ciphertext_len
            current_pos = source.tell()
            source.seek(tag_offset)
            tag = source.read(16)
            source.seek(current_pos)

            decryptor = Cipher(
                algorithms.AES(file_key), modes.GCM(nonce, tag)
            ).decryptor()
            metadata_bytes = decryptor.update(encrypted_metadata)
            try:
                original_rel = metadata_bytes.decode("utf-8")
            except UnicodeDecodeError:
                print(
                    "[SKIP] {} invalid metadata encoding".format(
                        os.path.basename(encrypted_file_path)
                    )
                )
                return

            restored_path = os.path.join(output_dir, original_rel)
            os.makedirs(os.path.dirname(restored_path), exist_ok=True)

            with open(restored_path, "wb", buffering=4 * 1024 * 1024) as restored:
                remaining = ciphertext_len
                while remaining > 0:
                    chunk = source.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    restored.write(decryptor.update(chunk))

            try:
                decryptor.finalize()
                print(
                    "[OK] {} -> {}".format(
                        os.path.basename(encrypted_file_path), original_rel
                    )
                )
            except Exception:
                if os.path.exists(restored_path):
                    os.remove(restored_path)
                print(
                    "[FAIL] {} tag verify failed".format(
                        os.path.basename(encrypted_file_path)
                    )
                )

    except Exception as error:  # pylint: disable=broad-except
        print(
            "[ERROR] decrypt failed for {}: {}".format(
                os.path.basename(encrypted_file_path), error
            )
        )


def batch_decrypt(vault_dir: str, output_dir: str, key: bytes) -> None:
    if not os.path.isdir(vault_dir):
        print(f"Vault dir not found: {vault_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for name in os.listdir(vault_dir):
        file_path = os.path.join(vault_dir, name)
        if not os.path.isfile(file_path):
            continue
        if name.startswith("."):
            continue
        decrypt_single_file(file_path, output_dir, key)
        count += 1

    print(f"Done. Processed {count} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decrypt encrypted recording files into original paths."
    )
    parser.add_argument("input_dir", help="Encrypted folder path")
    parser.add_argument("output_dir", help="Output folder path")
    args = parser.parse_args()

    if Cipher is None:
        print(f"Crypto unavailable: {CRYPTO_ERROR}")
        return

    key = _parse_key(os.getenv("ACE_ENCRYPTION_KEY"))
    if not key:
        return

    batch_decrypt(args.input_dir, args.output_dir, key)


if __name__ == "__main__":
    main()
