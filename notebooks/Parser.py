import urllib.parse

# Class to ingest a JSON Log Record and accomplish some action.

class Parser:
    DELIMETER = "|"
    def __init__(self, inlog) -> None:
        """
        inlog: give the JSON deserialized log object (it should be a dict())
        """
        assert type(inlog) == type(dict())

        self.inlog    = inlog
        self.keys     = set()
        self.element  = dict()
    
    def unwrap(self, log, key="", context="") -> None:
        if type(log) == type(dict()):
            for k in log.keys():
                self.keys.add(k)
                if context == "":
                    self.unwrap(log[k], key=k, context=k)
                else:
                    self.unwrap(log[k], key=k, context=context+"."+k)
        elif type(log) == type(list()):
            for el in log:
                self.unwrap(el, key=key, context=context)
        # Technicly, "Records" field should not exists anymore
        else:
            if key == "assumeRolePolicyDocument":
                log = urllib.parse.unquote(log).replace("\n", "").replace(" ", "")
            
            self.element[context] = log
            
    def parse_log(self) -> None:
        self.unwrap(self.inlog)

    def __str__(self) -> str:
        tmp = f"Parser(keys={len(self.keys)}) : "
        for k, v in self.element.items():
            tmp += f" '{k}'='{v}'; "
        return tmp

    def to_dict(self) -> dict:
        return self.element

    def to_csv(self) -> str:
        tmp = ""
        for k, v in self.element.items():
            tmp += f"{v}, "
        return tmp
    
    def create_csv_from_enum(self, enum) -> str:
        record = ""
        tmp_elements = self.element
        for e in enum:
            try:
                elem = tmp_elements.pop(e.value)
                record += str(elem) + self.DELIMETER
            except KeyError:
                record += "None" + self.DELIMETER
        # in this case discard other information => not good
        return record
    
    def return_only_first_level(self) -> [str]:
        results = []
        for k, el in enumerate(self.keys):
            if len(el.split(".")) == 1:
                results.append(el)

        return results 
        