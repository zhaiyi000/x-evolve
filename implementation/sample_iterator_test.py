from sample_iterator import SampleIterator
import os

if __name__ == "__main__":
    with open("sample.txt", "r", encoding="UTF-8") as file:
        lines = file.readlines()
        for line in lines:
            line_clean = line.strip().split(" ")
            store_folder_name = os.path.splitext(line_clean[0])[0]
            tune_sampler = SampleIterator(
                sample_name=line_clean[0],
                store_folder_name=store_folder_name,
                # regular=f"\[tunable\]\[((([\w.]+)\|)+([\w.]+))*\]",
                # regular=f"\[tunable\]\[([\w.]+)\|([\w.]+)*\]",
                # regular=f"\[tunable\]\[([\w.]+(?:\|[\w.]+)*)\]",
                regular=line_clean[1],
                split=f",",
            )
            tune_sampler.iterate_sample()
