 
LOCATIONS = ["l1", "l2", "l3"]
PARTS     = ["p1", "p2"]
 
TRANSPORT = {
    "tr1": {"capacity": 10, "cost": 9},
    "tr2": {"capacity": 8,  "cost": 15},
}
 
PART_SIZE = {
    "p1": 3,
    "p2": 2,
}
 
# negative values means that there is a demand while positive mean that there is a supply 
DEMAND_OFFER = {
    ("p1", "l1"):  6,
    ("p2", "l1"):  4,
    ("p1", "l3"): -6,
    ("p2", "l3"): -4,
}
 

ROUTES = {
    ("l1", "l2", "tr1"): {"dist": 3, "cost": 9},
    ("l1", "l2", "tr2"): {"dist": 3, "cost": 15},
    ("l2", "l3", "tr1"): {"dist": 2, "cost": 6},
    ("l2", "l3", "tr2"): {"dist": 2, "cost": 10},
}

MAX_QUANTITY = 10
 