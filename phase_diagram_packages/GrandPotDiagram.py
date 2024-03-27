'''
Created on Mar 22, 2023

@author: jiadongc
'''
from pymatgen.analysis.phase_diagram import PhaseDiagram,GrandPotPDEntry
from pymatgen.core.periodic_table import get_el_sp
from monty.json import MontyDecoder, MSONable

class GraPotPhaseDiagram(PhaseDiagram):
    def __init__(self, CPentries, chempots, elements=None, *, computed_data=None):
        """
        Standard constructor for grand potential phase diagram.

        Args:
            entries ([PDEntry]): A list of PDEntry-like objects having an
                energy, energy_per_atom and composition.
            chempots ({Element: float}): Specify the chemical potentials
                of the open elements.
            elements ([Element]): Optional list of elements in the phase
                diagram. If set to None, the elements are determined from
                the the entries themselves.
        """
        entries = [e.entry for e in CPentries]
        formEDict = {e.name:e.form_E * e.composition.num_atoms for e in CPentries}
        if elements is None:
            elements = {els for e in entries for els in e.composition.elements}

        self.chempots = {get_el_sp(el): u for el, u in chempots.items()}
        print(self.chempots)

        elements = set(elements).difference(self.chempots.keys())
        print(elements)

        all_entries = [
            GrandPotPDEntry(e, self.chempots) for e in entries if len(elements.intersection(e.composition.elements)) > 0
        ]
        
        '''change free energy to formation energy'''

        for entry in all_entries:
            e = entry.original_entry
            entry._energy = formEDict[e.name]
            
        super().__init__(all_entries, elements, computed_data=None)

    def __repr__(self):
        chemsys = "-".join([el.symbol for el in self.elements])
        chempots = ", ".join([f"u{el}={v}" for el, v in self.chempots.items()])

        output = [
            f"{chemsys} grand potential phase diagram with {chempots}",
            f"{len(self.stable_entries)} stable phases: ",
            ", ".join([entry.name for entry in self.stable_entries]),
        ]
        return "\n".join(output)

    def as_dict(self):
        """
        :return: MSONable dict
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "all_entries": [e.as_dict() for e in self.all_entries],
            "chempots": self.chempots,
            "elements": [e.as_dict() for e in self.elements],
        }

    @classmethod
    def from_dict(cls, d):
        """
        :param d: Dict representation
        :return: GrandPotentialPhaseDiagram
        """
        entries = MontyDecoder().process_decoded(d["all_entries"])
        elements = MontyDecoder().process_decoded(d["elements"])
        return cls(entries, d["chempots"], elements)