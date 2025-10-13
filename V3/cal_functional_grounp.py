from rdkit import Chem

# ✅ 定义常见官能团及其 SMARTS 模式
functional_groups = {
    "hydroxyl (-OH)": "[OX2H]",                            # 醇羟基
    "carbonyl (C=O)": "[CX3]=[OX1]",                      # 羰基
    "carboxyl (-COOH)": "C(=O)[OX2H1]",                   # 羧基
    "amine (-NH2)": "[NX3;H2]",                           # 胺基
    "amide (-CONH2)": "C(=O)N",                           # 酰胺
    "ester (-COOR)": "C(=O)O[C,H]",                       # 酯
    "ether (-O-)": "C-O-C",                               # 醚
    "nitro (-NO2)": "[NX3](=O)=O",                        # 硝基
    "halide (C-X)": "[C][F,Cl,Br,I]",                     # 卤代烃
    "alkene (C=C)": "C=C",                                # 烯烃
    "alkyne (C#C)": "C#C",                                # 炔烃
    "benzene ring": "c1ccccc1",                           # 苯环
    "boronate (B-O)": "B-O",                              # 硼氧键
}

# ✅ 输入 SMILES（可以是一组）
smiles_list = [
    "CC(=O)O",                    # 乙酸
    "CCO",                        # 乙醇
    "CC(=O)NC",                   # 乙酰胺
    "C1=CC=CC=C1",                # 苯
    "B(O)O",                      # 硼酸
    "CCBr",                       # 溴乙烷
]

# ✅ 遍历 SMILES，识别官能团
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    found_groups = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)
    print(f"SMILES: {smi}")
    print(f"  官能团: {found_groups if found_groups else '无匹配'}\n")
