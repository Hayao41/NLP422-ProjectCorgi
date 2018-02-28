''' preprocess semantic graph structure data from data file '''

import re
from semantic.SemanticStructure import SemanticGraphNode
from semantic.SemanticStructure import SemanticGraphEdge
from semantic.SemanticStructure import SemanticGraph
from semantic.SemanticStructure import SemanticGraphIterator
from utils import Constants
import utils.Utils as Utils
from collections import namedtuple

# split temp to [text], [pos], [index], [edge_type]
PATTERN = r'-> (.*?)/(.*?)\-(.*?) \((.*?)\)'

#=================== word, pos, relation dictionary =======================#
word = "I like this dog .".split() + "He loves that colorful pen .".split() + "He eventually went to New York City , and made records for King Records under the name Al Grant ( one in particular , ' Cabaret ' , appeared in the Variety magazine charts ) .".split()

pos = ["VBP",  "PRP", "NN", "DT", ".", "VBZ", "IN", "JJ"]+"PRP RB VBD TO NNP NNP NNP CC VBD NNS IN NNP NNPS IN DT NN NNP NNP CD IN JJ NN VBN IN DT NNP NN NNS ".split()

relation = ["nsubj", "dobj", "det", "punct", "amod", "punct", "root", "advmod", "nmod", "case", "compound", "cc", "conj", "acl", "dep"]

label = ["NO_ACTION", "CLAUSE_SPLIT"]

word2idx = Utils.make_dictionary(set(word), Constants.word_pad)

idx2word = {idx: word for word, idx in word2idx.items()}

pos2idx = Utils.make_dictionary(set(pos), Constants.pos_pad)

idx2pos = {idx: pos for pos, idx in pos2idx.items()}

rel2idx = Utils.make_dictionary(set(relation), Constants.rel_pad)

idx2rel = {idx: rel for rel, idx in rel2idx.items()}

label2idx = Utils.make_dictionary(label)

idx2label = {idx: label for label, idx in label2idx.items()}

DataTuple = namedtuple('DataTuple', ['indexedWords', 'graph'])
#==========================================================================#


def back_off(line_stack, indexedWords, 
            edgesOutgoing, edgesIncoming, 
            use_rel=True, rel2idx=None, 
            stop_offset=0):

    ''' 
    back off from leaf node(line_stack's top) to inst in line_stack, which's offset equals 
    to stop_offset then add word into indexedWords list along back off path and 
    create semantic graph edge then insert into edge table
    '''

    # back off from leaf node
    line_stack[-1][1].isLeaf = True

    while line_stack[-1][0] != stop_offset:
        
        # target node
        target = line_stack[-1][1]

        # target node's parent node
        source = line_stack[-2][1]

        # arc relation type
        relation = line_stack[-1][2]

        target.parent_index = source.sentence_index
        indexedWords.append(target)

        if relation not in rel2idx:
            rel_idx = rel2idx[Constants.UNK_REL]
        else:
            rel_idx = rel2idx[relation]

        curr_rel_idx = {True: rel_idx, False: None}

        edge = SemanticGraphEdge(
            source, 
            target, 
            relation, 
            rel_idx=curr_rel_idx[use_rel]
        )

        # add edge into outgoing edge table
        if source not in edgesOutgoing:
            edgesOutgoing[source] = [edge]
        else:
            edgesOutgoing[source].append(edge)

        # add edge into incoming edge table
        if target not in edgesIncoming:
            edgesIncoming[target] = [edge]
        else:
            edgesIncoming[target].append(edge)

        line_stack.pop()


def buildSemanticGraph(DependencyTreeStr, listLabel=None, 
                       use_word=True, use_pos=True,
                       use_rel=True, word2idx=None, 
                       pos2idx=None, rel2idx=None):
    
    ''' 
    re-build semantic graph with <SemanticGraph> structure from CoreNLP's dependency tree fromat str\n
    @Parameter\n
    CoreNLP's dependency tree fromat str(fromat:VALUE_TAG_INDEX)\n
    >>> -> root/VV-2 (root)\n
    >>>   -> leaf/NR-1 (nsubj)\n
    >>>   -> inner/VV-3 (conj)\n
    >>>     -> inner/VV-4 (ccomp)\n
    >>>       -> inner/NN-5 (dobj)\n
    @Return\n
    graph : <semantic.SemanticStructure.SemanticGraph>
    '''

    if use_word:
        assert word2idx is not None, "[Error]use_word set to 'True' but word2idx is None"
    if use_pos:
        assert pos2idx is not None, "[Error]use_pos set to 'True' but pos2idx is None"
    if use_rel:
        assert rel2idx is not None, "[Error]use_rel set to 'True' but rel2idx is None"
    
    line_stack = []
    indexedWords = []
    edgesOutgoing = {}
    edgesIncoming = {}
    
    listTemp = DependencyTreeStr.split("\n")
    listLines = []

    # split temp to [text], [pos], [index], [edge_type]
    for temp in listTemp:
        listLines.append(re.split(PATTERN, temp))

    #=========== tree root processing ================#

    # root node and root incoming arc attribute
    root_text = listLines[0][1]
    root_pos = listLines[0][2]
    root_senidx = int(listLines[0][3])
    root_incomarc_rel = listLines[0][4]

    if root_text not in word2idx:
        word_idx = word2idx[Constants.UNK_WORD]
    else:
        word_idx = word2idx[root_text]

    if root_pos not in pos2idx:
        pos_idx = pos2idx[Constants.UNK_POS]
    else:
        pos_idx = pos2idx[root_pos]

    if root_incomarc_rel not in rel2idx:
        rel_idx = rel2idx[Constants.UNK_REL]
    else:
        rel_idx = rel2idx[root_incomarc_rel]

    root_word_idx = {True: word_idx, False: None}
    root_pos_idx = {True: pos_idx, False: None}
    root_rel_idx = {True: rel_idx, False: None}

    # create root node
    root_node = SemanticGraphNode(
        text=root_text,
        pos=root_pos,
        sentence_index=root_senidx,
        word_idx=root_word_idx[use_word],
        pos_idx=root_pos_idx[use_pos],
        label=1
    )

    # create root's incoming edge(Padding edge)
    edgesIncoming[root_node] = [SemanticGraphEdge(
        source=None,
        target=root_node,
        relation=root_incomarc_rel,
        rel_idx=root_rel_idx[use_rel]
    )]

    indexedWords.append(root_node)

    line_stack.append((0, root_node, root_incomarc_rel))

    # DFS based str tree structure to build graph
    for line in listLines[1:]:
        
        # offset is very important to recognise back off point
        offset = len(line[0]) // 2
        text = line[1]
        pos = line[2]
        index = int(line[3])
        relation_type = line[4].split(":")[0]

        if text not in word2idx:
            word_idx = word2idx[Constants.UNK_WORD]
        else:
            word_idx = word2idx[text]

        if pos not in pos2idx:
            pos_idx = pos2idx[Constants.UNK_POS]
        else:
            pos_idx = pos2idx[pos]

        curr_word_idx = {True: word_idx, False: None}
        curr_pos_idx = {True: pos_idx, False: None}

        current_node = SemanticGraphNode(
                text=text,
                pos=pos,
                sentence_index=index,
                word_idx=curr_word_idx[use_word],
                pos_idx=curr_pos_idx[use_pos]
        )

        if current_node.sentence_index in listLabel:
            current_node.label = 1

        current = (
            offset, 
            current_node, 
            relation_type
        )

        # current line's offset is not stack top's plus 1 thus
        # stack top's is leaf node then back off util stack top's 
        # offset equals to current's minus 1(current node's parent)
        if (line_stack[-1][0] + 1) != current[0]:
            
            back_off(line_stack, indexedWords, 
                    edgesOutgoing, edgesIncoming,
                    use_rel, rel2idx,
                    stop_offset=(current[0] - 1))

        line_stack.append(current)

        # if current line is the last one, back off from current node 
        # util root node
        if line == listLines[-1]:
            
            back_off(line_stack, indexedWords, 
                    edgesOutgoing, edgesIncoming,
                    use_rel, rel2idx)

    # sort indexed words by sentence index
    indexedWords.sort(key=lambda indexedWord: indexedWord.sentence_index)

    # build semantic graph
    graph = SemanticGraph(root_node, edgesOutgoing, edgesIncoming, indexedWords)

    return graph
