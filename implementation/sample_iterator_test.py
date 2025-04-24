from sample_iterator import *
from redbaron import *


regular = "tunable\(\[(.*?)\]\)"
source = '''

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    """Returns priority with which we want to add item to each bin.

    This version uses a more sophisticated strategy that considers:
    1. Spatial fit: How well the item fits in the bin
    2. Future packing potential: How suitable the bin is for future items
    3. Bin utilization balance: Avoids creating bins that are too full or too empty

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Calculate basic fit metrics
    item_size = item
    bin_capacity = bins
    remaining_capacity = bin_capacity - item_size
    
    # 1. Spatial fit score: Better when item fits well and leaves reasonable space
    fit_quality = tunable([
        remaining_capacity / bin_capacity,  # Prefer bins where item fits but leaves some space
        np.exp(-np.abs(0.5 - remaining_capacity / bin_capacity)),  # Penalize bins that are too full or too empty
    ])
    
    # 2. Future packing potential: Bins that maintain good utilization balance
    # Penalize bins that are either too full or too empty
    utilization = 1 - remaining_capacity / bin_capacity
    utilization_penalty = np.exp(-tunable([2, 3, 4]) * np.abs(utilization - tunable([0.5, 0.6, 0.7])))
    
    # 3. Number of items in bin (we want to balance between spreading and consolidating)
    # Assuming bins is a 2D array where each row is a bin and last column is count of items
    num_items = bins[:, -1] if bins.ndim > 1 else np.zeros_like(bins)
    item_count_penalty = tunable([
        np.exp(-tunable([0.1, 0.2, 0.3]) * num_items),  # Prefer bins with fewer items
        1 / (1 + num_items),  # Simple inverse relationship
    ])
    
    # Combine all factors with tunable weights
    priority = tunable([0.4, 0.5, 0.6]) * fit_quality + \
              tunable([0.3, 0.4, 0.5]) * utilization_penalty + \
              tunable([0.2, 0.3, 0.4]) * item_count_penalty
    
    return priority


'''


# 首先需要安装libcst库: pip install libcst

import libcst as cst
from libcst.metadata import PositionProvider

def parse_tunables_with_comments(source_code):
    """解析源代码并保留注释/格式，找出tunable参数"""
    module = cst.parse_module(source_code)
    # print(module)
    tunables = []
    
    class TunableCollector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)
        
        def leave_Call(self, node):
            # 检查是否是tunable(...)调用
            if isinstance(node.func, cst.Name) and node.func.value == "tunable":
                # 检查第一个参数是否是列表
                if node.args and isinstance(node.args[0].value, cst.List):
                    list_elements = node.args[0].value.elements
                    pos = self.get_metadata(PositionProvider, node).start
                   
                    tunables.append({
                        "lineno": pos.line,
                        "column": pos.column,
                        "length": len(list_elements),
                        "node": node
                    })
    
    wrapper = cst.MetadataWrapper(module)
    collector = TunableCollector()
    wrapper.visit(collector)
    
    return tunables, module 

def replace_tunables_with_comments(module, tunables_info, replace_indices):
    """替换tunable参数并保持原始格式"""
    class TunableReplacer(cst.CSTTransformer):
        def __init__(self):
            self.current_index = 0
            
        def leave_Call(self, original_node, updated_node):
            nonlocal replace_indices
            if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "tunable":
                if self.current_index < len(replace_indices):
                    replace_idx = replace_indices[self.current_index]
                    self.current_index += 1
                    if updated_node.args and isinstance(updated_node.args[0].value, cst.List):
                        list_elements = updated_node.args[0].value.elements
                        # print(list_elements)
                        if 0 <= replace_idx < len(list_elements):
                            # return list_elements[replace_idx].value
                            # return list_elements[replace_idx].with_changes(comma=None)

                            # new_element = update`d_node.args[0].value.elements[0].with_changes(comma=None)

                            # new_whitespace = cst.ParenthesizedWhitespace(
                            #                     first_line=cst.TrailingWhitespace(
                            #                         whitespace=cst.SimpleWhitespace("  "),
                            #                         comment='1332221312',
                            #                         newline=None,
                            #                     ),
                            #                     indent=True,
                            #                     last_line=cst.SimpleWhitespace("    "),
                            #                 )
                            
                            # new_value = new_ele`ment.with_changes(whitespace_after=new_whitespace)
                            # list_elements[replace_idx].comma.with_changes(
                            #         whiteplace_after = cst.Comment(new_whitespace)
                            #         a = cst.Comma(s)
                            #     )
                            # if type(list_elements[replace_idx].comma.whitespace_before) == cst.SimpleWhitespace:
                            #     blank = list_elements[replace_idx].comma.whitespace_before.with_changes(
                            #         value='     '
                            #     )
                            # new_comma = list_elements[replace_idx].comma.with_changes(
                            #     whitespace_before = blank
                            # )
                            # print(list_elements)
                            return  list_elements[replace_idx].value
        
            return updated_node
    
    return module.visit(TunableReplacer())

# 比较标志
def is_in(str_1:str,str_2:str):
    if str_1 in str_2:
        _str_2 = str_2.split('=')
        for item in _str_2:
            clear_item = item.strip()
            if str_1 == clear_item:
                return True
    if 'tunable' in str_1:
        clear_str_1 = re.sub(regular,'tunable',str_1)
        _str_1 = clear_str_1.split('tunable')
        for item in _str_1:
            if item not in str_2:
                return False
        return True
    return False
        
    
# 示例用法
if __name__ == "__main__":

    # 解析参数
    tunables_info, original_module = parse_tunables_with_comments(source)
    print(f"找到 {len(tunables_info)} 个可调节参数:")
    for i, info in enumerate(tunables_info):
        print(f"参数组 {i+1} (行号:{info['lineno']}): 参数个数={info['length']}")

    # 替换所有tunable为对应列表的第一个元素
    modified_module = replace_tunables_with_comments(
        original_module,
        tunables_info,
        replace_indices=[0,0,0,1,0,0,0,0]
    )
    # RedBaron 找到所有形如'# 注释'的注释
    red = RedBaron(source)
    comments=[]
    for comment in red.find_all('comment'):
        comments.append({
            'text':comment.value.strip(),
        })
    # 找到注释所对应的行及行内标志信息
    lines = source.splitlines()
    for index,line in enumerate(lines):
        for item in comments:
            if item['text'] in line:
                item['sign'] = line.replace(item['text'],'').strip().strip(',')
                item['line'] = index
    # 仅保留所需注释
    comments_clear = []
    for item in comments:
        if len(item['sign']) != 0:
            comments_clear.append(item)
            print(item)
    # 注释回填
    mm_lines = modified_module.code.splitlines()
    final_code = []
    for index,line in enumerate(mm_lines):
        for item in comments_clear:
            if is_in(item['sign'],line) and item['line'] >= index:
                line = line + ' ' + item['text']
                item['line'] = -1
        final_code.append(line)
    # print(comments_clear)
                
    for item in final_code:
        print(item)
    
    # print("\n修改后的代码（保留注释和格式）:\n")
    # print(modified_module.code)
