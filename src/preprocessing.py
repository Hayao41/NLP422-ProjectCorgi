''' pre-processing for semantic graph structure data from data file or database'''

import re
from semantic.SemanticStructure import SemanticGraphNode
from semantic.SemanticStructure import SemanticGraphEdge
from semantic.SemanticStructure import SemanticGraph
from utils import Constants
import utils.Utils as Utils
from collections import namedtuple

# split temp to [text], [pos], [index], [edge_type]
PATTERN = r'-> (.*?)/(.*?)-(\d+) \((.*?)\)'
NUM_PATTERN = r'\d+'

# data namedTuple for wrapping sequence and semantic graph
DataTuple = namedtuple('DataTuple', ['indexedWords', 'graph'])


def buildAndSaveVocabDic(vocbList, vocabType, dic_path):
    
    ''' 
    build vocabulary dictionary then save into '*Dic.txt' file under 'src/vocabDic/' \n
    @Param:
    vocbList: vocabulary(word, pos, arc relation, action space)
    vocabType: word, pos, rel, act
    '''
    
    if "word" == vocabType or "pos" == vocabType or "rel" == vocabType or "act" == vocabType:
        padVocabType = {"word": Constants.word_pad, "pos": Constants.pos_pad, "rel": Constants.rel_pad, "act": {}}
        item2idx = Utils.make_dictionary(set(vocbList), padVocabType[vocabType])
        
        with open(dic_path + vocabType + "Dic.txt", "w", encoding="utf-8") as outputFile:
            outputFile.write(str(item2idx))
    else:
        raise Exception("no such vocab(word, pos, rel, act) type[{}]".format(vocabType))


def readDictionary(dic_path, mode='r', encoding='utf-8'):

    with open(dic_path, mode=mode, encoding=encoding) as file:
        dictionary = eval(file.read())

    return dictionary


def loadVocabDic(vocabTypes, dic_path):
    
    ''' 
    load builded vocabulary dictionary form 'src/vocabDic/*Dic.txt' file\n
    @Param:
    vocabTypes: vocabulary type list which you want to load from file, 
    4 types(word, pos, rel, act) at most
    '''
    
    vocabs = {"word": {}, "pos": {}, "rel": {}}

    if len(vocabTypes) > 3:
        raise Exception(
            "there are just 4 vocab types(word, pos, rel, act) but got [{}] types request".format(len(vocabTypes))
        )

    for vocabType in vocabTypes:
        if "word" == vocabType or "pos" == vocabType or "rel" == vocabType or "act" == vocabType:
            vocabs[vocabType] = readDictionary(
                dic_path + vocabType + "Dic.txt",
                mode="r",
                encoding="utf-8"
            )

        else:
            raise Exception("no such vocab(word, pos, rel) type[{}]".format(vocabType))
    return vocabs


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
        if target.sentence_index not in indexedWords:
            indexedWords[target.sentence_index] = target

        curr_rel_idx = lookUp(rel2idx, relation, use_item=use_rel)

        edge = SemanticGraphEdge(
            source, 
            target, 
            relation, 
            rel_idx=curr_rel_idx
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


def lookUp(item2idx, item, use_item=True):
    
    ''' look up dictionary to get item idx '''
    
    if use_item:
        if item not in item2idx:
            if item == '-LRB-':
                item_idx = item2idx['(']
            elif item == '-RRB-':
                item_idx = item2idx[')']
            else:
                item_idx = item2idx[Constants.UNKV]
                print(item)
        else:
            item_idx = item2idx[item]
    else:
        item_idx = None

    return item_idx

def reChecking(line, sid):
    
    if len(line) != 5:
        raise Exception("[RE Error] {} can't be spitted by pattern!".format(sid))


def buildSemanticGraph(DependencyTreeStr, listLabel=None, 
                       use_word=True, use_pos=True,
                       use_rel=True, word2idx=None, 
                       pos2idx=None, rel2idx=None,
                       sid="S#"):
    
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
    indexedWords = {}
    edgesOutgoing = {}
    edgesIncoming = {}
    
    listTemp = DependencyTreeStr.split("\n")[0:-1]
    listLines = []

    # split temp to [text], [pos], [index], [edge_type]
    for temp in listTemp:
        line = re.split(PATTERN, temp)
        reChecking(line, sid)
        if line[2].find("/") != -1:
            line[2] = line[2].split("/")[1]
        listLines.append(line)

    #=========== tree root processing ================#

    # root node and root incoming arc attribute
    root_text = listLines[0][1]
    root_pos = listLines[0][2]
    listLines[0][3] = re.findall(NUM_PATTERN, listLines[0][3])[0]
    root_senidx = int(listLines[0][3])
    root_incomarc_rel = listLines[0][4]

    root_word_idx = lookUp(word2idx, root_text, use_item=use_word)
    root_pos_idx = lookUp(pos2idx, root_pos, use_item=use_pos)
    root_rel_idx = lookUp(rel2idx, root_incomarc_rel, use_item=use_rel)

    # create root node
    root_node = SemanticGraphNode(
        text=root_text,
        pos=root_pos,
        sentence_index=root_senidx,
        word_idx=root_word_idx,
        pos_idx=root_pos_idx
    )

    if root_node.sentence_index in listLabel:
        root_node.label = 1

    # create root's incoming edge(Padding edge)
    edgesIncoming[root_node] = [SemanticGraphEdge(
        source=None,
        target=root_node,
        relation=root_incomarc_rel,
        rel_idx=root_rel_idx
    )]

    indexedWords[root_senidx] = root_node

    line_stack.append((0, root_node, root_incomarc_rel))

    # DFS based str tree structure to build graph
    for line in listLines[1:]:
        
        # offset is very important to recognise back off point
        offset = len(line[0]) // 2
        text = line[1]
        pos = line[2]
        line[3] = re.findall(NUM_PATTERN, line[3])[0]
        index = int(line[3])
        relation_type = line[4].split(":")[0]

        curr_word_idx = lookUp(word2idx, text, use_item=use_word)
        curr_pos_idx = lookUp(pos2idx, pos, use_item=use_pos)

        if index in indexedWords:
            current_node = indexedWords[index]
        else:
            current_node = SemanticGraphNode(
                    text=text,
                    pos=pos,
                    sentence_index=index,
                    word_idx=curr_word_idx,
                    pos_idx=curr_pos_idx
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
    indexedWords = list(indexedWords.values())
    indexedWords.sort(key=lambda indexedWord: indexedWord.sentence_index)

    # build semantic graph
    graph = SemanticGraph(
        root=root_node,
        edgesOutgoing=edgesOutgoing,
        edgesIncoming=edgesIncoming,
        indexedWords=indexedWords
    )

    return graph
