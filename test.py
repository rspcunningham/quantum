def hash_function(input_bits: list[int]) -> list[int]:

    assert len(input_bits) == 4
    extra_bits = [0] * 4
    hash_bits = [0] * 4

    extra_bits[0] = input_bits[0] ^ input_bits[2]
    extra_bits[1] = input_bits[1] ^ input_bits[3]
    extra_bits[2] = input_bits[0] ^ input_bits[1]
    extra_bits[3] = input_bits[2] ^ input_bits[3]

    hash_bits[0] = extra_bits[0] ^ input_bits[3]
    hash_bits[1] = extra_bits[1] ^ input_bits[0]
    hash_bits[2] = extra_bits[2] ^ input_bits[2]
    hash_bits[3] = extra_bits[3] ^ input_bits[1]

    return hash_bits


def create_lookup_table() -> dict[tuple[int, ...], tuple[int, ...]]:
    """
    Create a lookup table mapping all 16 possible 4-bit inputs to their hash outputs.

    Returns:
        Dictionary mapping input tuples to output tuples
    """
    lookup_table = {}

    # Generate all 16 possible 4-bit combinations (0000 to 1111)
    for i in range(16):
        # Convert integer to 4-bit binary representation
        input_bits = [
            (i >> 3) & 1,  # bit 3 (MSB)
            (i >> 2) & 1,  # bit 2
            (i >> 1) & 1,  # bit 1
            i & 1          # bit 0 (LSB)
        ]

        # Compute hash
        output_bits = hash_function(input_bits)

        # Store in lookup table using tuples as keys/values
        lookup_table[tuple(input_bits)] = tuple(output_bits)

    return lookup_table


# Example usage
if __name__ == "__main__":
    # Create and display the lookup table
    print("Complete Hash Function Lookup Table:")
    print("=" * 50)

    lookup_table = create_lookup_table()

    for input_bits, output_bits in sorted(lookup_table.items()):
        input_str = ''.join(map(str, input_bits))
        output_str = ''.join(map(str, output_bits))
        print(f"{input_str} -> {output_str}")
