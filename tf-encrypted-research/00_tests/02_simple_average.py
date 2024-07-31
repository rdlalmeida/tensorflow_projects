"""Example of a simple average using TF Encrypted."""
import time as t

start = t.time()
import tensorflow as tf
end = t.time()
print(f"Tensorflow lib took {end - start} seconds to load.")

start = t.time()
import tf_encrypted as tfe
end = t.time()
print(f"TF Encrypted took {end - start} seconds to load")


import argparse
from tf_encrypted.protocol import ABY3
from tf_encrypted.protocol import Pond
from tf_encrypted.protocol import SecureNN

if __name__ == "__main__":
    # Use this flag to determine the origin of the configuration file.
    config_origin = "local"
    
    parser = argparse.ArgumentParser(description="Train a TF Encrypted model")
    parser.add_argument(
        "--protocol",
        metavar="PROTOCOL",
        type=str,
        default="ABY3",
        help="MPC protocol TF Encrypted used",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="./tf-encrypted-research/02_config.json",
        help="path to configuration file",
    )
    parser.add_argument(
        "--precision",
        choices=["l", "h", "low", "high"],
        type=str,
        default="l",
        help="use 64 or 128 bits for computation",
    )

    args = parser.parse_args()

    # Set tfe config
    if config_origin != "local":
        # Config file was specified
        config_file = args.config
        config = tfe.RemoteConfig.load(config_file)
        config.connect_servers()
        tfe.set_config(config)
    else:
        # Always best practice to present all players to avoid invalid device errors
        config = tfe.LocalConfig(
            player_names=[
                "server0",
                "server1",
                "server2",
                "inputter-0",
                "inputter-1",
                "inputter-2",
                "inputter-3",
                "inputter-4",
                "result-receiver",
            ]
        )
        tfe.set_config(config)

    # Set tfe protocol
    tfe.set_protocol(globals()[args.protocol](fixedpoint_config=args.precision))

    @tfe.local_computation(name_scope="provide_input")
    def provide_input() -> tf.Tensor:
        # Pick random tensor to be averaged
        return tf.random.normal(shape=(10,))

    @tfe.local_computation(player_name="result-receiver", name_scope="receive_output")
    def receive_output(average: tf.Tensor):
        # Simply print average
        tf.print("Average: ", average)

    @tfe.local_computation(player_name="inputter-0", name_scope="inputter_input")
    def print_input(input: tf.Tensor, index: int):
        tf.print("Input #", str(index), " = ", input)

    # Get input from inputters as private values
    inputs = [
        provide_input(
            player_name="inputter-0"
        ),
        provide_input(
            player_name="inputter-1"
        ),
        provide_input(
            player_name="inputter-2"
        ),
        provide_input(
            player_name="inputter-3"
        ),
        provide_input(
            player_name="inputter-4"
        )
    ]

    i = 0
    for input in inputs:
        print_input(input, i)
        print("---------------------------------------------------")
        i += 1

    # Sum all inputs and divide by count
    result = tfe.add_n(inputs) / len(inputs)

    # Send result to receiver
    receive_output(result)