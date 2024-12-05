from os.path import expanduser
from data.pathways.gmt_pathway import get_KEGG_map
import os

## 下面这些代码单纯的就是为了确定文件所在的路径！
from os.path import join, realpath, dirname
BASE_PATH = dirname(realpath(__file__))
# print("测试一下这个BASE_PATH是谁", BASE_PATH)
DATA_PATH = join(BASE_PATH, '_database')


input_genes = ['AR', 'AKT', 'EGFR']
# # filename = expanduser('~/Data/pathways/MsigDB/c2.cp.kegg.v6.1.symbols.gmt')
# filename = '..\\_database\\pathways\\MsigDB\\c2.cp.kegg.v6.1.symbols.gmt'
# # filename = expanduser('./_database/pathways/MsigDB/c2.cp.kegg.v6.1.symbols.gmt')
#获取当前文件的目录
cur_path = os.path.abspath(os.path.dirname(__file__))
# 获取根目录
root_path = cur_path[:cur_path.find("\\pnet_prostate_paper-published_to_zenodo\\")+len("\\pnet_prostate_paper-published_to_zenodo\\")]
# print("测试的！", root_path + '_database/pathways/MsigDB/c2.cp.kegg.v6.1.symbols.gmt')
filename = root_path + '_database/pathways/MsigDB/c2.cp.kegg.v6.1.symbols.gmt'      ### 这里面是包含上百条通路，从左至右依次为：通路名、所在网址以及通路内部所包含的基因

mapp, genes, pathways = get_KEGG_map(input_genes, filename)      ### 利用该函数是可以从KEGG数据集中获取基因、通路以及两者之间关系的数据集的！
print(('genes', genes))     ### 在这这个是进行测试的，这个基因就三个
print(('pathways', pathways))
print(('mapp', mapp))       ## 这个就是最终的基因-通路关系矩阵，在这，这个基因很少只有三个'AR', 'AKT', 'EGFR'，而通路却非常多，其所形成二维矩阵，中间单元格表示当前通路与当前基因是否有关系，如果有关系则该位置取值1，否则取值0
