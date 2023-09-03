from pathlib import Path
import sys

if Path(sys.argv[1]).is_file():
    f = open(sys.argv[1],'r')
    lines = f.readlines()
    outfile = open('converted.nt','w')

    for l in lines:

        ents = [f'http://example.com/{x}' for x in l.strip().split('\t')]
        print(l)
        outfile.write(f'<{ents[0]}> <{ents[1]}> <{ents[2]}> .\n')
    outfile.close()

    # test:
    import lightrdf

    parser = lightrdf.Parser()

    for triple in parser.parse("/home/olli/gits/Better_Knowledge_Graph_Embeddings/codex/data/triples/codex-s/train.nt", base_iri=None):
        print(triple)
