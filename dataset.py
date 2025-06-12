from torch.utils.data import Dataset, DataLoader

class JediQADataset(Dataset):
    def __init__(self, dataset, tokenizer, prompt, instruct):
        super().__init__()
        self.data = dataset
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.system_instruct = instruct

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data.iloc[index]['CQ']
        answer = self.data.iloc[index]['Result NLP']
        user_prompt = self.prompt.format(question=question)
        message = [
            {"role": "system", "content": self.system_instruct},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"text": text, "answer":answer}
    
    

class JediQAOntoDataset(Dataset):
    def __init__(self, dataset, tokenizer, prompt, instruct, onto):
        super().__init__()
        self.data = dataset
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.system_instruct = instruct.format(ontology=onto)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data.iloc[index]['CQ']
        answer = self.data.iloc[index]['Result NLP']
        user_prompt = self.prompt.format(question=question)
        message = [
            {"role": "system", "content": self.system_instruct},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"text": text, "question":question, "answer":answer}

def get_dataloader(dataset, tokenizer, prompt, instruct, shuffle=False, batch_size=2):
    dataset_obj = JediQADataset(dataset, tokenizer, prompt, instruct)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle)


def get_dataloader_onto(dataset, tokenizer, prompt, instruct, onto, shuffle=False, batch_size=2):
    dataset_obj = JediQAOntoDataset(dataset, tokenizer, prompt, instruct, onto)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle)