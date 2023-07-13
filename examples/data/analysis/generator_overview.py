from collections import Counter

from matplotlib import pyplot as plt

from mlpf.data.loading.load_data import load_data, load_solved_from_tuple
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.bus_type import BusTypeIds
from mlpf.enumerations.generator_table import GeneratorTableIds

"""
Visualize the generation capabilities of the grid(looking only at a single ppc). 

* Look at bus types of generators to see if they're static or not.
* Look at the generator range i.e. the limits of each generator.
"""


def main():
    ppc = load_data("opf_ppcs", max_samples=1, load_sample_function=load_solved_from_tuple)[0]
    # ppc = load_data("solved_opf_ppcs", max_samples=1)[0]

    generator_bus_numbers = ppc["gen"][:, GeneratorTableIds.bus_number].astype(int)
    bus_types = ppc["bus"][:, BusTableIds.bus_type]
    generator_types = bus_types[generator_bus_numbers]

    print("Generator bus types: ", generator_types)

    counter = Counter(generator_types)

    labels = [BusTypeIds(key).name for key in counter.keys()]

    plt.figure()
    plt.title("Generator bus type distribution")
    plt.pie(counter.values(), labels=labels, autopct='%1.1f%%')

    plt.figure()
    plt.title("Generator active power allowed range")
    plt.vlines(x=generator_bus_numbers, ymin=ppc["gen"][:, GeneratorTableIds.min_active_power_MW], ymax=ppc["gen"][:, GeneratorTableIds.max_active_power_MW])
    plt.xlabel("bus #")
    plt.ylabel("$P_g^{range}$ [MW]")

    active_range_size = (ppc["gen"][:, GeneratorTableIds.max_active_power_MW] - ppc["gen"][:, GeneratorTableIds.min_active_power_MW])

    plt.scatter(x=generator_bus_numbers[active_range_size < 1e-3], y=active_range_size[active_range_size < 1e-3], marker='o', s=1)

    plt.figure()
    plt.title("Generator reactive power allowed range")
    plt.vlines(x=generator_bus_numbers, ymin=ppc["gen"][:, GeneratorTableIds.min_reactive_power_MVAr], ymax=ppc["gen"][:, GeneratorTableIds.max_reactive_power_MVAr])
    plt.xlabel("bus #")
    plt.ylabel("$Q_g^{range}$ [MVAr]")

    reactive_range_size = (ppc["gen"][:, GeneratorTableIds.max_reactive_power_MVAr] - ppc["gen"][:, GeneratorTableIds.min_reactive_power_MVAr])

    plt.scatter(x=generator_bus_numbers[reactive_range_size < 1e-3], y=reactive_range_size[reactive_range_size < 1e-3], marker='o', s=1)

    plt.show()


if __name__ == "__main__":
    main()
