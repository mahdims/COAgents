from os import path
import sys

sys.path.append(path.abspath(path.dirname(__file__)))
from ParameterReader import ParameterReader

class InstanceReader(ParameterReader):

    def __init__(self, file: str):
        super().__init__(None)
        self.file = file

    def data(self) -> bool:
        try:
            # List to hold customer information
            customer_info = []

            # Read file
            with open(self.file, 'r') as buffer:
                section = None
                for line_num, line in enumerate(buffer):
                    line = line.strip()

                    # Detecting section type
                    if line_num == 0 and line:
                        # First line is the instance name
                        self.instance = line.replace(" ", "")
                    elif "VEHICLE" in line:
                        section = "VEHICLE"
                    elif "CUSTOMER" in line:
                        section = "CUSTOMER"
                    elif section == "VEHICLE" and line_num == 4:
                        # Parsing vehicle information: NUMBER CAPACITY
                        tokens = line.split() if " " in line else line.split("\t")
                        if len(tokens) >= 2:
                            try:
                                self.nVehicles = int(tokens[0])
                                self.capacity = float(tokens[1])
                            except ValueError as e:
                                raise AssertionError(f"Error reading vehicle information.\n{e}")
                    elif section == "CUSTOMER" and line_num >= 9:
                        # Parsing customer information: CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME
                        tokens = line.split() if " " in line else line.split("\t")
                        if len(tokens) >= 7:
                            try:
                                customer_data = [
                                    int(tokens[0]),  # CUST NO.
                                    float(tokens[1]),  # XCOORD.
                                    float(tokens[2]),  # YCOORD.
                                    float(tokens[3]),  # DEMAND
                                    float(tokens[4]),  # READY TIME
                                    float(tokens[5]),  # DUE DATE
                                    float(tokens[6])   # SERVICE TIME
                                ]
                                customer_info.append(customer_data)
                            except ValueError as e:
                                raise AssertionError(f"Error reading customer information on line {line_num}.\n{e}")

            # Store information
            self.positions = {}
            self.service = {}
            self.demand = {}
            self.windows = {}

            # Populate class attributes with customer data
            for data in customer_info:
                index = data[0]
                self.positions[index] = [data[1], data[2]]
                self.demand[index] = data[3]
                self.windows[index] = [data[4], data[5]]
                self.service[index] = data[6]

            # Check if instance is valid
            if len(customer_info) < 1:
                return False

            self.nNodes = len(customer_info)
            self.depot = 0

        except IOError as e:
            raise AssertionError(f"Reading input file error.\n{e}")

        return True

    def info(self) -> str:
        if self.instance is None:
            return "no instance"
        output = f"file: {self.file}\ninstance: {self.instance}\nnumber of nodes: {self.nNodes}\nvehicles: {self.nVehicles}"
        return output

