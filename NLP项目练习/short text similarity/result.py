#Author ZhouJY

import GlobalParament
import utils


# 构建文档字典
def build_dict(file_dir):
    doc_dict = {}
    with open(file_dir, 'r', encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            line = utils.delete_r_n(line)
            id, content = line.split(',')
            doc_dict[id] = content

    return doc_dict


def sim_result(sim_out_dir, train_dict, test_dict, result_dir):
    f_writer = open(result_dir, 'w', encoding=GlobalParament.encoding)
    with open(sim_out_dir, 'r', encoding=GlobalParament.encoding) as sim_out_reader:
        for line in sim_out_reader:
            line = utils.delete_r_n(line)
            id, sim_ids = line.split(',')
            result = id + ',' + test_dict[id] + '\n' + '***最相似的10个文档***\n'
            result_train_ids = sim_ids.split()
            for id in result_train_ids:
                result = result + id + ',' + train_dict[id] + '\n'
            f_writer.write(result)
            f_writer.write('*************************************************\n')
            f_writer.flush()

    f_writer.close()



if __name__ == '__main__':
    train_dict = build_dict(GlobalParament.train_set_dir)
    test_dict = build_dict(GlobalParament.test_set_dir)
    sim_result(GlobalParament.result_output_path, train_dict, test_dict, GlobalParament.similarity_out_path)
