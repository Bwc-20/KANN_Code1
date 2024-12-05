import pandas as pd
from os.path import join
from config_path import PROSTATE_DATA_PATH

processed_dir = 'processed'
data_dir = 'raw_data'

processed_dir = join(PROSTATE_DATA_PATH, processed_dir)
data_dir = join(PROSTATE_DATA_PATH, data_dir)

### 此文件对下载的用于输入的癌症病人的数据进行进一步的处理（癌症状态用0、1代替，去除一些无关紧要的突变）



def prepare_design_matrix_crosstable():
    print('preparing mutations ...')

    filename = '41588_2018_78_MOESM4_ESM.txt'        ### 这个文件即为 补充表2 本队列中所有体细胞突变的完整列表（MAF文件）
    id_col = 'Tumor_Sample_Barcode'       ## 肿瘤样本条形码！
    print("现在要读的这个文件所在的路径是！", join(data_dir, filename))
    df = pd.read_csv(join(data_dir, filename), sep='\t', low_memory=False, skiprows=1)
    print('mutation distribution')
    print(df['Variant_Classification'].value_counts())

    if filter_silent_muts:
        df = df[df['Variant_Classification'] != 'Silent'].copy()         ### 在这是要滤除变异中的Silent哪些选项   在这Silent表示已删除的
    if filter_missense_muts:
        df = df[df['Variant_Classification'] != 'Missense_Mutation'].copy()         ### 在这 missense表示错义突变
    if filter_introns_muts:
        df = df[df['Variant_Classification'] != 'Intron'].copy()         ### 在此，Intron表示内含子

    # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
    exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
    if keep_important_only:
        df = df[~df['Variant_Classification'].isin(exclude)].copy()
    if truncating_only:
        include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
        df = df[df['Variant_Classification'].isin(include)].copy()
    df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                              aggfunc='count')
    df_table = df_table.fillna(0)
    total_numb_mutations = df_table.sum().sum()

    number_samples = df_table.shape[0]
    print('number of mutations', total_numb_mutations, total_numb_mutations / (number_samples + 0.0))
    filename = join(processed_dir, 'P1000_final_analysis_set_cross_' + ext + '.csv')
    df_table.to_csv(filename)


### 这个是来处理一下数据集中的那些结果标签（就是那些癌症的状态）    把最终的预测结果用0、1来表示
def prepare_response():
    print('preparing response ...')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(join(data_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
    response = pd.DataFrame()
    response['id'] = df['Patient.ID']
    response['response'] = df['Sample.Type']
    response['response'] = response['response'].replace('Metastasis', 1)
    response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    response.to_csv(join(processed_dir, 'response_paper.csv'), index=False)


def prepare_cnv():
    print('preparing copy number variants ...')
    filename = '41588_2018_78_MOESM10_ESM.txt'
    df = pd.read_csv(join(data_dir, filename), sep='\t', low_memory=False, skiprows=1, index_col=0)
    df = df.T           ## 在此 df.T表示对行列进行转置
    df = df.fillna(0.)
    filename = join(processed_dir, 'P1000_data_CNA_paper.csv')
    df.to_csv(filename)


def prepare_cnv_burden():
    print('preparing copy number burden(负荷) ...')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(join(data_dir, filename), skiprows=2, index_col=1)
    cnv = df['Fraction of genome altered']
    filename = join(processed_dir, 'P1000_data_CNA_burden.csv')
    cnv.to_frame().to_csv(filename)


# remove silent and intron mutations
filter_silent_muts = False
filter_missense_muts = False
filter_introns_muts = False
keep_important_only = True
truncating_only = False

ext = ""
if keep_important_only:
    ext = 'important_only'

if truncating_only:
    ext = 'truncating_only'

if filter_silent_muts:
    ext = "_no_silent"

if filter_missense_muts:
    ext = ext + "_no_missense"

if filter_introns_muts:
    ext = ext + "_no_introns"

prepare_design_matrix_crosstable()
prepare_cnv()
prepare_response()
prepare_cnv_burden()
print('Done')
